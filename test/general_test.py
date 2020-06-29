import torch
import numpy as np
import networkx as nx
from torch_geometric_temporal.nn.conv import GConvLSTM


def create_mock_data(number_of_nodes, edge_per_node, in_channels):

    graph = nx.watts_strogatz_graph(number_of_nodes, edge_per_node, 0.5)

    edges = torch.LongTensor(np.array([edge for edge in graph.edges()]).T)

    edge_weights = torch.FloatTensor(np.random.uniform(0, 1, (edges.shape[0], 1)))

    X = torch.FloatTensor(np.random.uniform(-1, 1, (number_of_nodes, in_channels)))

    return edges, edge_weights, X

def test_gconv_lstm_layer():
    """
    Testing the GConvLSTM Layer.
    """

    number_of_nodes = 100
    edge_per_node = 10
    in_channels = 64
    out_channels = 16
    K = 2

    edges, edge_weights, X = create_mock_data(number_of_nodes, edge_per_node, in_channels)

    layer = GConvLSTM(in_channels=in_channels, out_channels=out_channels,
                      K=K, number_of_nodes=number_of_nodes)

    assert layer.in_channels == in_channels
    assert layer.out_channels == out_channels
