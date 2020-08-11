import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.conv import GConvGRU


class Net(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(Net, self).__init__()
        self.recurrent_1 = GConvGRU(node_features, 32)
        self.recurrent_2 = GConvGRU(32, 16)
        self.linear = torch.nn.Linear(16, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.recurrent_1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.recurrent_2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


