import torch
from torch.nn import GRU
from torch_geometric.nn import GCNConv
from torch_geometric.nn import TopKPooling


class EvolveGCNH(torch.nn.Module):
    r"""An implementation of the Evolving Graph Convolutional Hidden Layer.
    For details see this paper: `"EvolveGCN: Evolving Graph Convolutional 
    Networks for Dynamic Graph." <https://arxiv.org/abs/1902.10191>`_

    Args:
        num_of_nodes (int): Number of vertices.
        in_channels (int): Number of filters.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
    """
    def __init__(self, num_of_nodes: int, in_channels: int, improved: bool=False,
                 cached: bool=False, normalize: bool=True, add_self_loops: bool=True):
        super(EvolveGCNH, self).__init__()

        self.num_of_nodes = num_of_nodes
        self.in_channels = in_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self._create_layers()


    def _create_layers(self):

        self.ratio = self.in_channels / self.num_of_nodes

        self.pooling_layer = TopKPooling(self.in_channels, self.ratio)

        self.recurrent_layer = GRU(input_size = self.in_channels,
                                   hidden_size = self.in_channels,
                                   num_layers = 1)


        self.conv_layer = GCNConv(in_channels = self.in_channels,
                                  out_channels = self.in_channels,
                                  improved = self.improved,
                                  cached = self.cached,
                                  normalize = self.normalize,
                                  add_self_loops = self.add_self_loops,
                                  bias = False)

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, 
                edge_weight: torch.FloatTensor=None) -> torch.FloatTensor:
        """
        Making a forward pass.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node embedding.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Float Tensor, optional)* - Edge weight vector.

        Return types:
            * **X** *(PyTorch Float Tensor)* - Output matrix for all nodes.
        """
        X_tilde = self.pooling_layer(X, edge_index)
        X_tilde = X_tilde[0][None, :, :]
        W = self.conv_layer.weight[None, :, :]
        X_tilde, W = self.recurrent_layer(X_tilde, W)
        self.conv_layer.weight = torch.nn.Parameter(W.squeeze())
        X = self.conv_layer(X, edge_index, edge_weight)
        return X
