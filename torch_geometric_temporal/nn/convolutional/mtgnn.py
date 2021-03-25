from __future__ import division

import numbers
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Linear(nn.Module):
    r"""An implementation of the linear layer, conducting 2D convolution.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks." 
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        bias (bool, optional): Whether to have bias. Default: True.
    """
    def __init__(self, c_in: int, c_out: int, bias: bool=True):
        super(Linear, self).__init__()
        self._mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(
            1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the linear layer.

        Arg types:
            * **X** (Pytorch Float Tensor) - Input tensor, with shape (batch_size, c_in, num_nodes, seq_len).

        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor, with shape (batch_size, c_out, num_nodes, seq_len).
        """
        return self._mlp(X)


class MixProp(nn.Module):
    r"""An implementation of the dynatic mix-hop propagation layer.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks." 
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        gdep (int): Depth of graph convolution.
        dropout (float): Dropout rate.
        alpha (float): Ratio of retaining the root nodes's original states, a value between 0 and 1.
    """

    def __init__(self, c_in: int, c_out: int, gdep: int, dropout: float, alpha: float):
        super(MixProp, self).__init__()
        self._mlp = Linear((gdep+1)*c_in, c_out)
        self._gdep = gdep
        self._dropout = dropout
        self._alpha = alpha

    def forward(self, X: torch.FloatTensor, A: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of mix-hop propagation.

        Arg types:
            * **X** (Pytorch Float Tensor) - Input feature Tensor, with shape (batch_size, c_in, num_nodes, seq_len).
            * **A** (PyTorch Float Tensor) - Adjacency matrix, with shape (num_nodes, num_nodes).

        Return types:
            * **H_0** (PyTorch Float Tensor) - Hidden representation for all nodes, with shape (batch_size, c_out, num_nodes, seq_len).
        """
        A = A + torch.eye(A.size(0)).to(X.device)  # add self-loops
        d = A.sum(1)
        H = X
        H_0 = X
        A = A / d.view(-1, 1)
        for i in range(self._gdep):
            H = self._alpha*X + (1 - self._alpha) * torch.einsum('ncwl,vw->ncvl', (H, A))
            H_0 = torch.cat((H_0,H), dim=1)
        del i
        H_0 = self._mlp(H_0)
        return H_0


class DilatedInception(nn.Module):
    r"""An implementation of the dilated inception layer.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks." 
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        kernel_set (list of int): List of kernel sizes.
        dilated_factor (int, optional): Dilation factor. Default: 2.
    """
    def __init__(self, c_in: int, c_out: int, kernel_set: list, dilation_factor: int=2):
        super(DilatedInception, self).__init__()
        self._time_conv = nn.ModuleList()
        self._kernel_set = kernel_set
        c_out = int(c_out/len(self._kernel_set))
        for kern in self._kernel_set:
            self._time_conv.append(nn.Conv2d(c_in, c_out, (1, kern),
                                        dilation=(1, dilation_factor)))

    def forward(self, X_in: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of dilated inception.

        Arg types:
            * **X_in** (Pytorch Float Tensor) - Input feature Tensor, with shape (batch_size, c_in, num_nodes, seq_len).

        Return types:
            * **X** (PyTorch Float Tensor) - Hidden representation for all nodes, 
            with shape (batch_size, c_out, num_nodes, seq_len-6).
        """
        X = []
        for i in range(len(self._kernel_set)):
            X.append(self._time_conv[i](X_in))
        for i in range(len(self._kernel_set)):
            X[i] = X[i][..., -X[-1].size(3):]
        X = torch.cat(X, dim=1)
        return X


class GraphConstructor(nn.Module):
    r"""An implementation of the graph learning layer to construct an adjacency matrix.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks." 
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        nnodes (int): Number of nodes in the graph.
        k (int): Number of largest values to consider in constructing the neighbourhood of a node (pick the "nearest" k nodes).
        dim (int): Dimension of the node embedding.
        alpha (float, optional): Tanh alpha for generating adjacency matrix, alpha controls the saturation rate, default 3.
        xd (int, optional): Static feature dimension, default None.
    """

    def __init__(self, nnodes: int, k: int, dim: int, alpha: float=3, xd: Optional[int]=None):
        super(GraphConstructor, self).__init__()
        if xd is not None:
            self._static_feature_dim = xd
            self._linear1 = nn.Linear(xd, dim)
            self._linear2 = nn.Linear(xd, dim)
        else:
            self._embedding1 = nn.Embedding(nnodes, dim)
            self._embedding2 = nn.Embedding(nnodes, dim)
            self._linear1 = nn.Linear(dim, dim)
            self._linear2 = nn.Linear(dim, dim)

        self._k = k
        self._alpha = alpha

    def forward(self, idx: torch.LongTensor, static_feat: Optional[torch.FloatTensor]=None) -> torch.FloatTensor:
        """
        Making a forward pass to construct an adjacency matrix from node embeddings.

        Arg types:
            * **idx** (Pytorch Long Tensor) - Input indices, a permutation of the number of nodes, default None (no permutation).
            * **static_feat** (Pytorch Float Tensor, optional) - Static feature, default None.
        Return types:
            * **adj** (PyTorch Float Tensor) - Adjacency matrix constructed from node embeddings.
        """

        if static_feat is None:
            nodevec1 = self._embedding1(idx)
            nodevec2 = self._embedding2(idx)
        else:
            assert static_feat.shape[1] == self._static_feature_dim
            nodevec1 = static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self._alpha*self._linear1(nodevec1))
        nodevec2 = torch.tanh(self._alpha*self._linear2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - \
            torch.mm(nodevec2, nodevec1.transpose(1, 0))
        A = F.relu(torch.tanh(self._alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(A.device)
        mask.fill_(float('0'))
        s1, t1 = A.topk(self._k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        A = A*mask
        return A


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight',
                     'bias', 'eps', 'elementwise_affine']
    r"""An implementation of the layer normalization layer.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks." 
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        normalized_shape (int): Input shape from an expected input of size.
        eps (float, optional): Value added to the denominator for numerical stability. Default: 1e-5.
        elementwise_affine (bool, optional): Whether to conduct elementwise affine transformation or not. Default: True.
    """
    def __init__(self, normalized_shape: int, eps: float=1e-5, elementwise_affine: bool=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self._normlizationalized_shape = tuple(normalized_shape)
        self._eps = eps
        self._elementwise_affine = elementwise_affine
        if self._elementwise_affine:
            self._weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self._bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self._reset_parameters()

    def _reset_parameters(self):
        if self._elementwise_affine:
            init.ones_(self._weight)
            init.zeros_(self._bias)

    def forward(self, X: torch.FloatTensor, idx: torch.LongTensor) -> torch.FloatTensor:
        """
        Making a forward pass of layer normalization.

        Arg types:
            * **X** (Pytorch Float Tensor) - Input tensor, with shape (batch_size, feature_dim, num_nodes, seq_len).
            * **idx** (Pytorch Long Tensor) - Input indices.
            
        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor, with shape (batch_size, feature_dim, num_nodes, seq_len).
        """
        if self._elementwise_affine:
            return F.layer_norm(X, tuple(X.shape[1:]), self._weight[:, idx, :], self._bias[:, idx, :], self._eps)
        else:
            return F.layer_norm(X, tuple(X.shape[1:]), self._weight, self._bias, self._eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class MTGNN(nn.Module):
    r"""An implementation of the Multivariate Time Series Forecasting Graph Neural Networks.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks." 
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        gcn_true (bool) : Whether to add graph convolution layer.
        build_adj (bool) : Whether to construct adaptive adjacency matrix.
        gcn_depth (int) : Graph convolution depth.
        num_nodes (int) : Number of nodes in the graph.
        kernel_set (list of int): List of kernel sizes.
        dropout (float, optional) : Droupout rate, default 0.3.
        subgraph_size (int, optional) : Size of subgraph, default 20.
        node_dim (int, optional) : Dimension of nodes, default 40.
        dilation_exponential (int, optional) : Dilation exponential, default 1.
        conv_channels (int, optional) : Convolution channels, default 32.
        residual_channels (int, optional) : Residual channels, default 32.
        skip_channels (int, optional) : Skip channels, default 64.
        end_channels (int, optional): End channels, default 128.
        seq_length (int, optional) : Length of input sequence, default 12.
        in_dim (int, optional) : Input dimension, default 2.
        out_dim (int, optional) : Output dimension, default 12.
        layers (int, optional) : Number of layers, default 3.
        propalpha (float, optional) : Prop alpha, ratio of retaining the root nodes's original states in mix-hop propagation, a value between 0 and 1, default 0.05.
        tanhalpha (float, optional) : Tanh alpha for generating adjacency matrix, alpha controls the saturation rate, default 3.
        layer_norm_affline (bool, optional) : Whether to do elementwise affine in Layer Normalization, default True.
    """

    def __init__(self, gcn_true: bool, build_adj: bool, gcn_depth: int, num_nodes: int,  kernel_set: list, dropout: float=0.3, 
    subgraph_size: int=20, node_dim: int=40, dilation_exponential: int=1, conv_channels: int=32, residual_channels: int=32, 
    skip_channels: int=64, end_channels: int=128, seq_length: int=12, in_dim: int=2, out_dim: int=12, layers: int=3, 
    propalpha: float=0.05, tanhalpha: float=3, layer_norm_affline: bool=True):
        super(MTGNN, self).__init__()
        self._gcn_true = gcn_true
        self._build_adj_true = build_adj
        self._num_nodes = num_nodes
        self._dropout = dropout
        self._filter_convs = nn.ModuleList()
        self._gate_convs = nn.ModuleList()
        self._residual_convs = nn.ModuleList()
        self._skip_convs = nn.ModuleList()
        self._mixprop_conv1 = nn.ModuleList()
        self._mixprop_conv2 = nn.ModuleList()
        self._normlization = nn.ModuleList()
        self._start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self._graph_constructor = GraphConstructor(
            num_nodes, subgraph_size, node_dim, alpha=tanhalpha)

        self._seq_length = seq_length
        kernel_size = 7
        if dilation_exponential > 1:
            self._receptive_field = int(
                1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self._receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1, layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self._filter_convs.append(DilatedInception(
                    residual_channels, conv_channels, kernel_set=kernel_set, dilation_factor=new_dilation))
                self._gate_convs.append(DilatedInception(
                    residual_channels, conv_channels, kernel_set=kernel_set, dilation_factor=new_dilation))
                self._residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                if self._seq_length > self._receptive_field:
                    self._skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, self._seq_length-rf_size_j+1)))
                else:
                    self._skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, self._receptive_field-rf_size_j+1)))

                if self._gcn_true:
                    self._mixprop_conv1.append(
                        MixProp(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self._mixprop_conv2.append(
                        MixProp(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self._seq_length > self._receptive_field:
                    self._normlization.append(LayerNorm(
                        (residual_channels, num_nodes, self._seq_length - rf_size_j + 1), elementwise_affine=layer_norm_affline))
                else:
                    self._normlization.append(LayerNorm(
                        (residual_channels, num_nodes, self._receptive_field - rf_size_j + 1), elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self._layers = layers
        self._end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self._end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)
        if self._seq_length > self._receptive_field:
            self._skip_conv_0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(
                1, self._seq_length), bias=True)
            self._skip_conv_E = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(
                1, self._seq_length-self._receptive_field+1), bias=True)

        else:
            self._skip_conv_0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(
                1, self._receptive_field), bias=True)
            self._skip_conv_E = nn.Conv2d(in_channels=residual_channels,
                                   out_channels=skip_channels, kernel_size=(1, 1), bias=True)

        self._idx = torch.arange(self._num_nodes)

    def forward(self, X_in: torch.FloatTensor, Tilde_A: Optional[torch.FloatTensor]=None, idx: Optional[torch.LongTensor]=None, static_feat: Optional[torch.FloatTensor]=None) -> torch.FloatTensor:
        """
        Making a forward pass of MTGNN.

        Arg types:
            * X_in (PyTorch Float Tensor) - Input sequence, 
            with shape (batch size, input dimension, number of nodes, input sequence length).
            * Tilde_A (Pytorch Float Tensor, optional) - Predefined adjacency matrix, default None.
            * idx (Pytorch Long Tensor, optional) - Input indices, a permutation of the number of nodes, default None (no permutation).
            * static_feat (Pytorch Float Tensor, optional) - Static feature, default None.

        Return types:
            * x (PyTorch Float Tensor) - Output sequence for prediction, 
            with shape (batch size, input sequence length, number of nodes, 1).
        """
        seq_len = X_in.size(3)
        assert seq_len == self._seq_length, 'input sequence length not equal to preset sequence length'

        if self._seq_length < self._receptive_field:
            X_in = nn.functional.pad(
                X_in, (self._receptive_field-self._seq_length, 0, 0, 0))

        if self._gcn_true:
            if self._build_adj_true:
                if idx is None:
                    adp = self._graph_constructor(self._idx.to(X_in.device),
                                  static_feat=static_feat)
                else:
                    adp = self._graph_constructor(idx, static_feat=static_feat)
            else:
                adp = Tilde_A

        X = self._start_conv(X_in)
        skip = self._skip_conv_0(
            F.dropout(X_in, self._dropout, training=self.training))
        for i in range(self._layers):
            residual = X
            filter = self._filter_convs[i](X)
            filter = torch.tanh(filter)
            gate = self._gate_convs[i](X)
            gate = torch.sigmoid(gate)
            X = filter * gate
            X = F.dropout(X, self._dropout, training=self.training)
            s = X
            s = self._skip_convs[i](s)
            skip = s + skip
            if self._gcn_true:
                X = self._mixprop_conv1[i](X, adp)+self._mixprop_conv2[i](X,
                                                          adp.transpose(1, 0))
            else:
                X = self._residual_convs[i](X)

            X = X + residual[:, :, :, -X.size(3):]
            if idx is None:
                X = self._normlization[i](X, self._idx)
            else:
                X = self._normlization[i](X, idx)

        skip = self._skip_conv_E(X) + skip
        X = F.relu(skip)
        X = F.relu(self._end_conv_1(X))
        X = self._end_conv_2(X)
        return X
