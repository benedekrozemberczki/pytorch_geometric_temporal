import torch
import numpy as np
import networkx as nx
import torch.nn.functional as F
from torch_geometric_temporal.nn.conv import GConvGRU


def create_mock_data(number_of_nodes, edge_per_node, in_channels):
    """
    Creating a mock feature matrix and edge index.
    """
    graph = nx.watts_strogatz_graph(number_of_nodes, edge_per_node, 0.5)
    edge_index = torch.LongTensor(np.array([edge for edge in graph.edges()]).T)
    X = torch.FloatTensor(np.random.uniform(-1, 1, (number_of_nodes, in_channels)))
    return X, edge_index


def create_mock_states(number_of_nodes, out_channels):
    """
    Creating mock hidden and cell states. 
    """
    H = torch.FloatTensor(np.random.uniform(-1, 1, (number_of_nodes, in_channels)))
    C = torch.FloatTensor(np.random.uniform(-1, 1, (number_of_nodes, in_channels)))
    return H, C


def create_mock_edge_weight(edge_index):
    """
    Creating a mock edge weight tensor.
    """
    return torch.FloatTensor(np.random.uniform(0, 1, (edge_index.shape[1])))


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


