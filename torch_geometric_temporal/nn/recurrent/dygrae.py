import torch
import torch.nn as nn
from torch_geometric.nn import GatedGraphConv
from torch.nn.utils import clip_grad_norm_

class DyGrEncoder(torch.nn.Module):
    r"""An implementation of the integrated Gated Graph Convolution Long Short
    Term Memory Layer. For details see this paper: `"Predictive Temporal Embedding
    of Dynamic Graphs." <https://ieeexplore.ieee.org/document/9073186>`_

    Args:
        conv_out_channels (int): Number of output channels for the GGCN.
        conv_num_layers (int): Number of Gated Graph Convolutions.
        conv_aggr (str): Aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
        lstm_out_channels (int): Number of LSTM channels.
        lstm_num_layers (int): Number of neurons in LSTM.
    """

    def __init__(
        self,
        conv_out_channels: int,
        conv_num_layers: int,
        conv_aggr: str,
        lstm_out_channels: int,
        lstm_num_layers: int,
    ):
        super(DyGrEncoder, self).__init__()
        assert conv_aggr in ["mean", "add", "max"], "Wrong aggregator."
        self.conv_out_channels = conv_out_channels
        self.conv_num_layers = conv_num_layers
        self.conv_aggr = conv_aggr
        self.lstm_out_channels = lstm_out_channels
        self.lstm_num_layers = lstm_num_layers
        self._create_layers()

    def _create_layers(self):
        self.conv_layer = GatedGraphConv(
            out_channels=self.conv_out_channels,
            num_layers=self.conv_num_layers,
            aggr=self.conv_aggr,
            bias=True,
        )

        self.recurrent_layer = nn.LSTMCell(
            input_size=self.conv_out_channels,
            hidden_size=self.lstm_out_channels,
        )

        self.dropout = nn.Dropout(0.5)  # Adjust dropout rate as needed

        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        H_tilde = self.conv_layer(X, edge_index, edge_weight)
        H_tilde = self.dropout(H_tilde)

        batch_size = H_tilde.size(0)

        if H is None and C is None:
            H = torch.zeros(batch_size, self.lstm_out_channels).to(X.device)
            C = torch.zeros(batch_size, self.lstm_out_channels).to(X.device)

        H_out = []
        C_out = []

        for i in range(batch_size):
            H_i, C_i = self.recurrent_layer(H_tilde[i], (H[i], C[i]))
            H_out.append(H_i)
            C_out.append(C_i)

        H_out = torch.stack(H_out)
        C_out = torch.stack(C_out)

        return H_tilde, H_out, C_out

# Example usage with data loader and GPU support
model = DyGrEncoder(conv_out_channels, conv_num_layers, conv_aggr, lstm_out_channels, lstm_num_layers)
model = model.to('cuda')  # Move model to GPU

# Example data loader
dataset = YourDataset(...)  # Replace with your own dataset
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch_data in data_loader:
        batch_data = batch_data.to('cuda')  # Move data to GPU
        optimizer.zero_grad()
        output, h_out, c_out = model(batch_data.X, batch_data.edge_index, batch_data.edge_weight)
        loss = compute_loss(output, batch_data.y)  # Replace with your own loss computation
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients to prevent explosion
        optimizer.step()
