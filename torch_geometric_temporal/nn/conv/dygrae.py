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
                 lstm_out_channels: int,
                 num_relations: int, num_bases: int):
        super(DyGrEncoder, self).__init__()

        self.conv_out_channels = conv_out_channels
        self.lstm_out_channels = lstm_out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self._create_layers()


    def _create_layers(self):
        self._create_input_gate_layers()
        self._create_forget_gate_layers()
        self._create_cell_state_layers()
        self._create_output_gate_layers()


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
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_type, H, C)
        F = self._calculate_forget_gate(X, edge_index, edge_type, H, C)
        C = self._calculate_cell_state(X, edge_index, edge_type, H, C, I, F)
        O = self._calculate_output_gate(X, edge_index, edge_type, H, C)
        H = self._calculate_hidden_state(O, C)
        return H, C
