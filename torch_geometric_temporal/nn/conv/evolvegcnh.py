import torch
from torch.nn import GRU
from torch_geometric.nn import GCNConv
from torch_geometric.nn import TopKPooling


class EvolveGCNH(torch.nn.Module):
    r"""An implementation of the integrated Gated Graph Convolution Long Short
    Term Memory Layer. For details see this paper: `"Predictive Temporal Embedding
    of Dynamic Graphs." <https://ieeexplore.ieee.org/document/9073186>`_

    Args:
        conv_out_channels (int): Number of neurons in GGCN.
        conv_num_layers (int): Number of Gated Graph Convolutions.
        conv_aggr (str): Aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
        lstm_out_channels (int): Number of LSTM channels.
        lstm_num_layers (int): Number of neurons in LSTM.
    """
    def __init__(self, num_of_nodes: int, in_channels: int):
        super(EvolveGCNH, self).__init__()

        self.num_of_nodes = num_of_nodes
        self.in_channels = in_channels
        self._create_layers()


    def _create_layers(self):

        self.recurrent_layer = GRU(input_size = self.in_channels,
                                   hidden_size = self.in_channels,
                                   num_layers = 1)


        self.conv_layer = GCNConv(in_channels = self.in_channels,
                                  out_channels = self.out_channels,
                                  bias = False)

    def forward(self, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor=None,
                H: torch.FloatTensor=None, W: torch.FloatTensor=None) -> torch.FloatTensor:
        """
        Making a forward pass. If the hidden state and cell state matrices are 
        not present when the forward pass is called these are initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Float Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.

        Return types:
            * **H_tilde** *(PyTorch Float Tensor)* - Output matrix for all nodes.
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        if H is None and W is None:
            pass
        elif H is not None and W is not None:
            pass
        else:
            raise ValueError("Invalid hidden state and cell matrices.")
        H_tilde = H_tilde.squeeze()
        H = H.squeeze()
        C = C.squeeze()
        return H_tilde, H, C
