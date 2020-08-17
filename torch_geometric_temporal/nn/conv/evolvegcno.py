import torch
from torch.nn import LSTM
from torch_geometric.nn import GCNConv


class EvolveGCNO(torch.nn.Module):
    r"""An implementation of the Evolving Graph Convolutional without Hidden Layer.
    For details see this paper: `"EvolveGCN: Evolving Graph Convolutional 
    Networks for Dynamic Graph." <https://arxiv.org/abs/1902.10191>`_

    Args:
        in_channels (int): Number of filters.
    """
    def __init__(self, in_channels: int):
        super(EvolveGCNO, self).__init__()

        self.in_channels = in_channels
        self._create_layers()


    def _create_layers(self):

        self.recurrent_layer = LSTM(input_size = self.in_channels,
                                    hidden_size = self.in_channels,
                                    num_layers = 1)


        self.conv_layer = GCNConv(in_channels = self.in_channels,
                                  out_channels = self.in_channels,
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
        W = self.conv_layer.weight[None, :, :]
        W, _ = self.recurrent_layer(W)
        self.conv_layer.weight = torch.nn.Parameter(W.squeeze())
        X = self.conv_layer(X, edge_index, edge_weight)
        return X
