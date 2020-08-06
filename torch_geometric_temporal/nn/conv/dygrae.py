import torch
from torch.nn import LSTM
from torch_geometric.nn import GatedGraphConv

class DyGrEncoder(torch.nn.Module):
    r"""An implementation of the integrated Gated Graph Convolution Long Short
    Term Memory Layer. For details see this paper: `"Predictive Temporal Embedding
    of Dynamic Graphs." <https://ieeexplore.ieee.org/document/9073186>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases.
    """
    def __init__(self, conv_channels: int, conv_num_layers: int=2, conv_aggr: str="add",
                 lstm_out_channels: int, lstm_num_layers: int=1):
        super(DyGrEncoder, self).__init__()

        self.conv_out_channels = conv_out_channels
        self.conv_num_layers = conv_num_layers
        self.conv_aggr = conv_aggr
        self.lstm_out_channels = lstm_out_channels
        self.lstm_num_layers = lstm_num_layers
        self._create_layers()


    def _create_layers(self):
        self.conv_layer = GatedGraphConv()
        self.recurrent_layer = LSTM()


    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor,
                edge_weight: torch.LongTensor, H: torch.FloatTensor=None, C: torch.FloatTensor=None):
        """
        Making a forward pass. If the hidden state and cell state matrices are 
        not present when the forward pass is called these are initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_type** *(PyTorch Long Tensor)* - Edge type vector.
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes (optional).
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes (optional).

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        H = self.conv_layer(X, edge_index, edge_weight)
        return H
