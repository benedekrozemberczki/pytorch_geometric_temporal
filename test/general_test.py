import torch
import numpy as np
import networkx as nx
from torch_geometric_temporal.nn.conv import GConvLSTM, GConvGRU, GCLSTM


def create_mock_data(number_of_nodes, edge_per_node, in_channels):

    graph = nx.watts_strogatz_graph(number_of_nodes, edge_per_node, 0.5)

    edge_index = torch.LongTensor(np.array([edge for edge in graph.edges()]).T)

    edge_weight = torch.FloatTensor(np.random.uniform(0, 1, (edge_index.shape[1])))

    X = torch.FloatTensor(np.random.uniform(-1, 1, (number_of_nodes, in_channels)))

    return X, edge_index, edge_weight


def _create_mock_states(number_of_nodes, out_channels):
    H = torch.FloatTensor(np.random.uniform(-1, 1, (number_of_nodes, in_channels)))
    C = torch.FloatTensor(np.random.uniform(-1, 1, (number_of_nodes, in_channels)))
    return H, C


def test_gconv_lstm_layer():
    """
    Testing the GConvLSTM Layer.
    """
    number_of_nodes = 100
    edge_per_node = 10
    in_channels = 64
    out_channels = 16
    K = 2

    X, edge_index, edge_weight = create_mock_data(number_of_nodes, edge_per_node, in_channels)

    layer = GConvLSTM(in_channels=in_channels, out_channels=out_channels, K=K)


    H, C = layer(X, edge_index)

    assert H.shape == (number_of_nodes, out_channels)
    assert C.shape == (number_of_nodes, out_channels)
    

    H, C = layer(X, edge_index, edge_weight)

    assert H.shape == (number_of_nodes, out_channels)
    assert C.shape == (number_of_nodes, out_channels)

    H, C = layer(X, edge_index, edge_weight, H, C)

    assert H.shape == (number_of_nodes, out_channels)
    assert C.shape == (number_of_nodes, out_channels)


def test_gconv_gru_layer():
    """
    Testing the GConvGRU Layer.
    """
    number_of_nodes = 100
    edge_per_node = 10
    in_channels = 64
    out_channels = 16
    K = 2

    X, edge_index, edge_weight = create_mock_data(number_of_nodes, edge_per_node, in_channels)

    layer = GConvGRU(in_channels=in_channels, out_channels=out_channels, K=K)


    H = layer(X, edge_index)

    assert H.shape == (number_of_nodes, out_channels)

    H = layer(X, edge_index, edge_weight)

    assert H.shape == (number_of_nodes, out_channels)

    H = layer(X, edge_index, edge_weight, H)

    assert H.shape == (number_of_nodes, out_channels)


def test_gc_lstm_layer():
    """
    Testing the GCLSTM Layer.
    """
    number_of_nodes = 100
    edge_per_node = 10
    in_channels = 64
    out_channels = 16
    K = 2

    X, edge_index, edge_weight = create_mock_data(number_of_nodes, edge_per_node, in_channels)

    layer = GCLSTM(in_channels=in_channels, out_channels=out_channels, K=K)

    H, C = layer(X, edge_index)

    assert H.shape == (number_of_nodes, out_channels)
    assert C.shape == (number_of_nodes, out_channels)
    

    H, C = layer(X, edge_index, edge_weight)

    assert H.shape == (number_of_nodes, out_channels)
    assert C.shape == (number_of_nodes, out_channels)

    H, C = layer(X, edge_index, edge_weight, H, C)

    assert H.shape == (number_of_nodes, out_channels)
    assert C.shape == (number_of_nodes, out_channels)
