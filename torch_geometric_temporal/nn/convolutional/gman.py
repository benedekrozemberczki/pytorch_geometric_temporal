import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Callable


class conv2d_(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, kernel_size: int, stride: tuple = (1, 1),
                 padding: str = 'SAME', use_bias: bool = True, activation: Callable[[torch.FloatTensor], torch.FloatTensor] = F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self._padding_size = math.ceil(kernel_size)
        else:
            self._padding_size = [0, 0]
        self._conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                               padding=0, bias=use_bias)
        self._batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self._conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self._conv.bias)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self._padding_size[1], self._padding_size[1],
                       self._padding_size[0], self._padding_size[0]]))
        x = self._conv(x)
        x = self._batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims: int, units: Union[int, tuple], activations: Union[Callable[[torch.FloatTensor], torch.FloatTensor], list], bn_decay: float, use_bias: bool = True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self._convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for conv in self._convs:
            x = conv(x)
        return x


class STEmbedding(nn.Module):
    r"""An implementation of the spatial-temporal embedding block.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction." <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        D (int) : dimension of output.
        bn_decay (float): batch normalization momentum.
    """

    def __init__(self, D: int, bn_decay: float):
        super(STEmbedding, self).__init__()
        self._FC_se = FC(
            input_dims=[D, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)

        self._FC_te = FC(
            input_dims=[295, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)  # input_dims = time step per day + days per week=288+7=295

    def forward(self, SE: torch.FloatTensor, TE: torch.FloatTensor, T: int = 288) -> torch.FloatTensor:
        """
        Making a forward pass of the spatial-temporal embedding.

        Arg types:
            * **SE** (PyTorch Float Tensor) - spatial embedding, with shape (num_nodes, D).
            * **TE** (Pytorch Float Tensor) - temporal embedding, with shape (batch_size, num_his + num_pred, 2).(dayofweek, timeofday)
            * **T** (int, optional) - num of time steps in one day, default 288.

        Return types:
            * output (PyTorch Float Tensor) - spatial-temporal embedding, with shape (batch_size, num_his + num_pred, num_nodes, D).
        """
        # spatial embedding
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self._FC_se(SE)
        # temporal embedding
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 288, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        TE = TE.unsqueeze(dim=2)
        TE = self._FC_te(TE)
        del dayofweek, timeofday
        return SE + TE


class SpatialAttention(nn.Module):
    r"""An implementation of the spatial attention mechanism.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction." <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        K (int) : number of attention heads.
        d (int) : dimension of each attention head outputs.
        bn_decay (float): batch normalization momentum.
    """

    def __init__(self, K: int, d: int, bn_decay: float):
        super(SpatialAttention, self).__init__()
        D = K * d
        self._d = d
        self._K = K
        self._FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                        bn_decay=bn_decay)
        self._FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                        bn_decay=bn_decay)
        self._FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                        bn_decay=bn_decay)
        self._FC = FC(input_dims=D, units=D, activations=F.relu,
                      bn_decay=bn_decay)

    def forward(self, X: torch.FloatTensor, STE: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the spatial attention mechanism.

        Arg types:
            * **X** (PyTorch Float Tensor) - input sequence, with shape (batch_size, num_step, num_nodes, K*d), where num_step can be num_his or num_pred.
            * **STE** (Pytorch Float Tensor) - spatial-temporal embedding, with shape (batch_size, num_step, num_nodes, K*d).

        Return types:
            * **X** (PyTorch Float Tensor) - spatial attention scores, with shape (batch_size, num_step, num_nodes, K*d).
        """
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_step, num_nodes, K * d]
        query = self._FC_q(X)
        key = self._FC_k(X)
        value = self._FC_v(X)
        # [K * batch_size, num_step, num_nodes, d]
        query = torch.cat(torch.split(query, self._K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self._K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self._K, dim=-1), dim=0)
        # [K * batch_size, num_step, num_nodes, num_nodes]
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self._d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_nodes, D]
        X = torch.matmul(attention, value)
        # orginal K, change to batch_size
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self._FC(X)
        del query, key, value, attention
        return X


class TemporalAttention(nn.Module):
    r"""An implementation of the temporal attention mechanism.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction." <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        K (int) : number of attention heads.
        d (int) : dimension of each attention head outputs.
        bn_decay (float): batch normalization momentum.
        mask (bool, optional): whether to mask attention score.
    """

    def __init__(self, K: int, d: int, bn_decay: float, mask: bool = True):
        super(TemporalAttention, self).__init__()
        D = K * d
        self._d = d
        self._K = K
        self._mask = mask
        self._FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                        bn_decay=bn_decay)
        self._FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                        bn_decay=bn_decay)
        self._FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                        bn_decay=bn_decay)
        self._FC = FC(input_dims=D, units=D, activations=F.relu,
                      bn_decay=bn_decay)

    def forward(self, X: torch.FloatTensor, STE: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the temporal attention mechanism.

        Arg types:
            * **X** (PyTorch Float Tensor) - input sequence, with shape (batch_size, num_step, num_nodes, K*d), where num_step can be num_his or num_pred.
            * **STE** (Pytorch Float Tensor) - spatial-temporal embedding, with shape (batch_size, num_step, num_nodes, K*d).

        Return types:
            * **X** (PyTorch Float Tensor) - temporal attention scores, with shape (batch_size, num_step, num_nodes, K*d).
        """
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_step, num_nodes, K * d]
        query = self._FC_q(X)
        key = self._FC_k(X)
        value = self._FC_v(X)
        # [K * batch_size, num_step, num_nodes, d]
        query = torch.cat(torch.split(query, self._K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self._K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self._K, dim=-1), dim=0)
        # query: [K * batch_size, num_nodes, num_step, d]
        # key:   [K * batch_size, num_nodes, d, num_step]
        # value: [K * batch_size, num_nodes, num_step, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * batch_size, num_nodes, num_step, num_step]
        attention = torch.matmul(query, key)
        attention /= (self._d ** 0.5)
        # mask attention score
        if self._mask:
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_nodes = X.shape[2]
            mask = torch.ones(num_step, num_step)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self._K * batch_size, num_nodes, 1, 1)
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, -2 ** 15 + 1)
        # softmax
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_nodes, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        # orginal K, change to batch_size
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self._FC(X)
        del query, key, value, attention
        return X


class GatedFusion(nn.Module):
    r"""An implementation of the gated fusion mechanism.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction." <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        D (int) : dimension of output.
        bn_decay (float): batch normalization momentum.
    """

    def __init__(self, D: int, bn_decay: float):
        super(GatedFusion, self).__init__()
        self._FC_xs = FC(input_dims=D, units=D, activations=None,
                         bn_decay=bn_decay, use_bias=False)
        self._FC_xt = FC(input_dims=D, units=D, activations=None,
                         bn_decay=bn_decay, use_bias=True)
        self._FC_h = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None],
                        bn_decay=bn_decay)

    def forward(self, HS: torch.FloatTensor, HT: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the gated fusion mechanism.

        Arg types:
            * **HS** (PyTorch Float Tensor) - spatial attention scores, with shape (batch_size, num_step, num_nodes, D), where num_step can be num_his or num_pred.
            * **HT** (Pytorch Float Tensor) - temporal attention scores, with shape (batch_size, num_step, num_nodes, D).

        Return types:
            * **H** (PyTorch Float Tensor) - spatial-temporal attention scores, with shape (batch_size, num_step, num_nodes, D).
        """
        XS = self._FC_xs(HS)
        XT = self._FC_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self._FC_h(H)
        del XS, XT, z
        return H


class STAttBlock(nn.Module):
    r"""An implementation of the spatial-temporal attention block, with spatial attention and temporal attention followed by gated fusion.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction." <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        K (int) : number of attention heads.
        d (int) : dimension of each attention head outputs.
        bn_decay (float): batch normalization momentum.
        mask (bool, optional): whether to mask attention score in temporal attention.
    """

    def __init__(self, K: int, d: int, bn_decay: float, mask: bool = False):
        super(STAttBlock, self).__init__()
        self._spatial_attention = SpatialAttention(K, d, bn_decay)
        self._temporal_attention = TemporalAttention(K, d, bn_decay, mask=mask)
        self._gated_fusion = GatedFusion(K * d, bn_decay)

    def forward(self, X: torch.FloatTensor, STE: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the spatial-temporal attention block.

        Arg types:
            * **X** (PyTorch Float Tensor) - input sequence, with shape (batch_size, num_step, num_nodes, K*d), where num_step can be num_his or num_pred.
            * **STE** (Pytorch Float Tensor) - spatial-temporal embedding, with shape (batch_size, num_step, num_nodes, K*d).

        Return types:
            * **X** (PyTorch Float Tensor) - attention scores, with shape (batch_size, num_step, num_nodes, K*d).
        """
        HS = self._spatial_attention(X, STE)
        HT = self._temporal_attention(X, STE)
        H = self._gated_fusion(HS, HT)
        del HS, HT
        X = torch.add(X, H)
        return X


class TransformAttention(nn.Module):
    r"""An implementation of the tranform attention mechanism.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction." <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        K (int) : number of attention heads.
        d (int) : dimension of each attention head outputs.
        bn_decay (float): batch normalization momentum.
    """

    def __init__(self, K: int, d: int, bn_decay: float):
        super(TransformAttention, self).__init__()
        D = K * d
        self._K = K
        self._d = d
        self._FC_q = FC(input_dims=D, units=D, activations=F.relu,
                        bn_decay=bn_decay)
        self._FC_k = FC(input_dims=D, units=D, activations=F.relu,
                        bn_decay=bn_decay)
        self._FC_v = FC(input_dims=D, units=D, activations=F.relu,
                        bn_decay=bn_decay)
        self._FC = FC(input_dims=D, units=D, activations=F.relu,
                      bn_decay=bn_decay)

    def forward(self, X: torch.FloatTensor, STE_his: torch.FloatTensor, STE_pred: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the transform attention layer.

        Arg types:
            * **X** (PyTorch Float Tensor) - input sequence, with shape (batch_size, num_his, num_nodes, K*d).
            * **STE_his** (Pytorch Float Tensor) - spatial-temporal embedding for history, with shape (batch_size, num_his, num_nodes, K*d).
            * **STE_pred** (Pytorch Float Tensor) - spatial-temporal embedding for prediction, with shape (batch_size, num_pred, num_nodes, K*d).

        Return types:
            * **X** (PyTorch Float Tensor) - output sequence for prediction, with shape (batch_size, num_pred, num_nodes, K*d).
        """
        batch_size = X.shape[0]
        # [batch_size, num_step, num_nodes, K * d]
        query = self._FC_q(STE_pred)
        key = self._FC_k(STE_his)
        value = self._FC_v(X)
        # [K * batch_size, num_step, num_nodes, d]
        query = torch.cat(torch.split(query, self._K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self._K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self._K, dim=-1), dim=0)
        # query: [K * batch_size, num_nodes, num_pred, d]
        # key:   [K * batch_size, num_nodes, d, num_his]
        # value: [K * batch_size, num_nodes, num_his, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * batch_size, num_nodes, num_pred, num_his]
        attention = torch.matmul(query, key)
        attention /= (self._d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_pred, num_nodes, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self._FC(X)
        del query, key, value, attention
        return X


class GMAN(nn.Module):
    r"""An implementation of GMAN.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction." <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        num_his (int): number of history steps.
        num_pred (int): number of prediction steps.
        T (int) : one day is divided into T steps.
        L (int) : number of STAtt blocks in the encoder/decoder.
        K (int) : number of attention heads.
        d (int) : dimension of each attention head outputs.
        bn_decay (float): batch normalization momentum.
    """

    def __init__(self, L: int, K: int, d: int, num_his: int, bn_decay: float):
        super(GMAN, self).__init__()
        D = K * d
        self._num_his = num_his
        self._st_embedding = STEmbedding(D, bn_decay)
        self._st_att_block1 = nn.ModuleList(
            [STAttBlock(K, d, bn_decay) for _ in range(L)])
        self._st_att_block2 = nn.ModuleList(
            [STAttBlock(K, d, bn_decay) for _ in range(L)])
        self._transform_attention = TransformAttention(K, d, bn_decay)
        self._FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None],
                        bn_decay=bn_decay)
        self._FC_2 = FC(input_dims=[D, D], units=[D, 1], activations=[F.relu, None],
                        bn_decay=bn_decay)

    def forward(self, X: torch.FloatTensor, SE: torch.FloatTensor, TE: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of GMAN.

        Arg types:
            * **X** (PyTorch Float Tensor) - input sequence, with shape (batch_size, num_hist, num of nodes).
            * **SE** (Pytorch Float Tensor) ： spatial embedding, with shape (numbed of nodes, K * d).
            * **TE** (Pytorch Float Tensor) - temporal embedding, with shape (batch_size, num_his + num_pred, 2) (time-of-day, day-of-week).

        Return types:
            * **X** (PyTorch Float Tensor) - output sequence for prediction, with shape (batch_size, num_pred, num of nodes).
        """
        # input
        X = torch.unsqueeze(X, -1)
        X = self._FC_1(X)
        # STE
        STE = self._st_embedding(SE, TE)
        STE_his = STE[:, :self._num_his]
        STE_pred = STE[:, self._num_his:]
        # encoder
        for net in self._st_att_block1:
            X = net(X, STE_his)
        # transAtt
        X = self._transform_attention(X, STE_his, STE_pred)
        # decoder
        for net in self._st_att_block2:
            X = net(X, STE_pred)
        # output
        X = self._FC_2(X)
        del STE, STE_his, STE_pred
        return torch.squeeze(X, 3)
