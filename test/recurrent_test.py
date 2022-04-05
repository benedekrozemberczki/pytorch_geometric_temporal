import torch
import numpy as np
import networkx as nx
from torch_geometric_temporal.nn.recurrent import GConvLSTM, GConvGRU, DCRNN, AGCRN
from torch_geometric_temporal.nn.recurrent import GCLSTM, LRGCN, DyGrEncoder
from torch_geometric_temporal.nn.recurrent import (
    EvolveGCNH,
    EvolveGCNO,
    TGCN,
    A3TGCN,
    A3TGCN2,
    MPNNLSTM,
)


def create_mock_data(number_of_nodes, edge_per_node, in_channels):
    """
    Creating a mock feature matrix and edge index.
    """
    graph = nx.watts_strogatz_graph(number_of_nodes, edge_per_node, 0.5)
    edge_index = torch.LongTensor(np.array([edge for edge in graph.edges()]).T)
    X = torch.FloatTensor(np.random.uniform(-1, 1, (number_of_nodes, in_channels)))
    return X, edge_index




def create_mock_attention_data(number_of_nodes, edge_per_node, in_channels, periods):
    """
    Creating a mock stacked feature matrix and edge index.
    """
    graph = nx.watts_strogatz_graph(number_of_nodes, edge_per_node, 0.5)
    edge_index = torch.LongTensor(np.array([edge for edge in graph.edges()]).T)
    X = torch.FloatTensor(
        np.random.uniform(-1, 1, (number_of_nodes, in_channels, periods))
    )
    return X, edge_index



def create_mock_attention_batch_data(number_of_nodes, edge_per_node, in_channels, periods, batch_size):
    """
    Creating a mock stacked feature matrix in batches and edge index.
    """
    graph = nx.watts_strogatz_graph(number_of_nodes, edge_per_node, 0.5)
    edge_index = torch.LongTensor(np.array([edge for edge in graph.edges()]).T)
    X = torch.FloatTensor(
        np.random.uniform(-1, 1, (batch_size, number_of_nodes, in_channels, periods))
    )
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


def create_mock_edge_relations(edge_index, num_relations):
    """
    Creating a mock relation type tensor.
    """
    return torch.LongTensor(
        np.random.choice(num_relations, edge_index.shape[1], replace=True)
    )


def test_gconv_lstm_layer():
    """
    Testing the GConvLSTM Layer.
    """
    number_of_nodes = 100
    edge_per_node = 10
    in_channels = 64
    out_channels = 16
    K = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, edge_index = create_mock_data(number_of_nodes, edge_per_node, in_channels)
    X = X.to(device)
    edge_index = edge_index.to(device)
    edge_weight = create_mock_edge_weight(edge_index).to(device)

    layer = GConvLSTM(in_channels=in_channels, out_channels=out_channels, K=K).to(
        device
    )

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, edge_index = create_mock_data(number_of_nodes, edge_per_node, in_channels)
    X = X.to(device)
    edge_index = edge_index.to(device)
    edge_weight = create_mock_edge_weight(edge_index).to(device)

    layer = GConvGRU(in_channels=in_channels, out_channels=out_channels, K=K).to(device)

    H = layer(X, edge_index)

    assert H.shape == (number_of_nodes, out_channels)

    H = layer(X, edge_index, edge_weight)

    assert H.shape == (number_of_nodes, out_channels)

    H = layer(X, edge_index, edge_weight, H)

    assert H.shape == (number_of_nodes, out_channels)


def test_mpnn_lstm_layer():
    """
    Testing the MPNN LSTM Layer.
    """
    number_of_nodes = 100
    edge_per_node = 10
    in_channels = 64
    hidden_size = 32
    window = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, edge_index = create_mock_data(number_of_nodes, edge_per_node, in_channels)
    X = X.to(device)
    edge_index = edge_index.to(device)
    edge_weight = create_mock_edge_weight(edge_index).to(device)

    layer = MPNNLSTM(
        in_channels=in_channels,
        hidden_size=hidden_size,
        num_nodes=number_of_nodes,
        window=window,
        dropout=0.5,
    ).to(device)

    H = layer(X, edge_index, edge_weight)

    assert H.shape == (number_of_nodes, 2 * hidden_size + in_channels + window - 1)


def test_tgcn_layer():
    """
    Testing the T-GCN Layer.
    """
    number_of_nodes = 100
    edge_per_node = 10
    in_channels = 64
    out_channels = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, edge_index = create_mock_data(number_of_nodes, edge_per_node, in_channels)
    X = X.to(device)
    edge_index = edge_index.to(device)
    edge_weight = create_mock_edge_weight(edge_index).to(device)

    layer = TGCN(in_channels=in_channels, out_channels=out_channels).to(device)

    H = layer(X, edge_index)

    assert H.shape == (number_of_nodes, out_channels)

    H = layer(X, edge_index, edge_weight)

    assert H.shape == (number_of_nodes, out_channels)

    H = layer(X, edge_index, edge_weight, H)

    assert H.shape == (number_of_nodes, out_channels)


def test_a3tgcn_layer():
    """
    Testing the A3TGCN Layer.
    """
    number_of_nodes = 100
    edge_per_node = 10
    in_channels = 64
    out_channels = 16
    periods = 7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, edge_index = create_mock_attention_data(
        number_of_nodes, edge_per_node, in_channels, periods
    )
    X = X.to(device)
    edge_index = edge_index.to(device)
    edge_weight = create_mock_edge_weight(edge_index).to(device)

    layer = A3TGCN(
        in_channels=in_channels, out_channels=out_channels, periods=periods
    ).to(device)

    H = layer(X, edge_index)

    assert H.shape == (number_of_nodes, out_channels)

    H = layer(X, edge_index, edge_weight)

    assert H.shape == (number_of_nodes, out_channels)

    H = layer(X, edge_index, edge_weight, H)

    assert H.shape == (number_of_nodes, out_channels)


def test_a3tgcn2_layer():
    """
    Testing the A3TGCN2 Layer by adding a batch index.
    """
    number_of_nodes = 100
    edge_per_node = 10
    in_channels = 64
    out_channels = 16
    periods = 7
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, edge_index = create_mock_attention_batch_data(
        number_of_nodes, edge_per_node, in_channels, periods, batch_size
    )
    X = X.to(device)
    edge_index = edge_index.to(device)
    edge_weight = create_mock_edge_weight(edge_index).to(device)

    layer = A3TGCN2(
        in_channels=in_channels, out_channels=out_channels, periods=periods, batch_size=batch_size
    ).to(device)

    H = layer(X, edge_index)

    assert H.shape == (batch_size, number_of_nodes, out_channels)

    H = layer(X, edge_index, edge_weight)

    assert H.shape == (batch_size, number_of_nodes, out_channels)

    H = layer(X, edge_index, edge_weight, H)

    assert H.shape == (batch_size, number_of_nodes, out_channels)
    

def test_dcrnn_layer():
    """
    Testing the DCRNN Layer.
    """
    number_of_nodes = 100
    edge_per_node = 10
    in_channels = 64
    out_channels = 16
    K = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, edge_index = create_mock_data(number_of_nodes, edge_per_node, in_channels)
    X = X.to(device)
    edge_index = edge_index.to(device)
    edge_weight = create_mock_edge_weight(edge_index).to(device)

    layer = DCRNN(in_channels=in_channels, out_channels=out_channels, K=K).to(device)

    H = layer(X, edge_index)

    assert H.shape == (number_of_nodes, out_channels)

    H = layer(X, edge_index, edge_weight)

    assert H.shape == (number_of_nodes, out_channels)

    H = layer(X, edge_index, edge_weight, H)

    assert H.shape == (number_of_nodes, out_channels)

    layer = DCRNN(in_channels=in_channels, out_channels=out_channels, K=3).to(device)

    H = layer(X, edge_index, edge_weight, H)

    assert H.shape == (number_of_nodes, out_channels)

    H = layer(X, edge_index, edge_weight)

    assert H.shape == (number_of_nodes, out_channels)

    H = layer(X, edge_index, edge_weight, H)

    assert H.shape == (number_of_nodes, out_channels)


def test_agcrn_layer():
    """
    Testing the AGCRN Layer.
    """
    number_of_nodes = 100
    edge_per_node = 10
    in_channels = 64
    out_channels = 16
    embedding_dimensions = 32
    K = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, edge_index = create_mock_data(number_of_nodes, edge_per_node, in_channels)
    X = X.view(-1, number_of_nodes, in_channels)
    X = X.to(device)
    E = torch.nn.Parameter(
        torch.randn(number_of_nodes, embedding_dimensions), requires_grad=True
    ).to(device)

    layer = AGCRN(
        number_of_nodes=number_of_nodes,
        in_channels=in_channels,
        out_channels=out_channels,
        K=K,
        embedding_dimensions=embedding_dimensions,
    ).to(device)

    H = layer(X, E)

    assert H.shape == (1, number_of_nodes, out_channels)

    H = layer(X, E, H)

    assert H.shape == (1, number_of_nodes, out_channels)

    layer = AGCRN(
        number_of_nodes=number_of_nodes,
        in_channels=in_channels,
        out_channels=out_channels,
        K=3,
        embedding_dimensions=embedding_dimensions,
    ).to(device)

    H = layer(X, E)

    assert H.shape == (1, number_of_nodes, out_channels)

    H = layer(X, E, H)

    assert H.shape == (1, number_of_nodes, out_channels)


def test_gc_lstm_layer():
    """
    Testing the GCLSTM Layer.
    """
    number_of_nodes = 100
    edge_per_node = 10
    in_channels = 64
    out_channels = 16
    K = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, edge_index = create_mock_data(number_of_nodes, edge_per_node, in_channels)
    X = X.to(device)
    edge_index = edge_index.to(device)
    edge_weight = create_mock_edge_weight(edge_index).to(device)

    layer = GCLSTM(in_channels=in_channels, out_channels=out_channels, K=K).to(device)

    H, C = layer(X, edge_index)

    assert H.shape == (number_of_nodes, out_channels)
    assert C.shape == (number_of_nodes, out_channels)

    H, C = layer(X, edge_index, edge_weight)

    assert H.shape == (number_of_nodes, out_channels)
    assert C.shape == (number_of_nodes, out_channels)

    H, C = layer(X, edge_index, edge_weight, H, C)

    assert H.shape == (number_of_nodes, out_channels)
    assert C.shape == (number_of_nodes, out_channels)


def test_lrgcn_layer():
    """
    Testing the LRGCN Layer.
    """
    number_of_nodes = 100
    edge_per_node = 10
    in_channels = 64
    out_channels = 16
    num_relations = 5
    num_bases = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, edge_index = create_mock_data(number_of_nodes, edge_per_node, in_channels)
    X = X.to(device)
    edge_index = edge_index.to(device)
    edge_relations = create_mock_edge_relations(edge_index, num_relations).to(device)

    layer = LRGCN(
        in_channels=in_channels,
        out_channels=out_channels,
        num_relations=num_relations,
        num_bases=num_bases,
    ).to(device)

    H, C = layer(X, edge_index, edge_relations)

    assert H.shape == (number_of_nodes, out_channels)
    assert C.shape == (number_of_nodes, out_channels)

    H, C = layer(X, edge_index, edge_relations, H, C)

    assert H.shape == (number_of_nodes, out_channels)
    assert C.shape == (number_of_nodes, out_channels)


def test_dygrencoder_layer():
    """
    Testing the DyGrEncoder Layer.
    """
    number_of_nodes = 100
    edge_per_node = 10
    in_channels = 8

    conv_out_channels = 16
    conv_num_layers = 3
    conv_aggr = "add"
    lstm_out_channels = 8
    lstm_num_layers = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, edge_index = create_mock_data(number_of_nodes, edge_per_node, in_channels)
    X = X.to(device)
    edge_index = edge_index.to(device)
    edge_weight = create_mock_edge_weight(edge_index).to(device)

    layer = DyGrEncoder(
        conv_out_channels=conv_out_channels,
        conv_num_layers=conv_num_layers,
        conv_aggr=conv_aggr,
        lstm_out_channels=lstm_out_channels,
        lstm_num_layers=lstm_num_layers,
    ).to(device)

    H_tilde, H, C = layer(X, edge_index)

    assert H_tilde.shape == (number_of_nodes, lstm_out_channels)
    assert H.shape == (number_of_nodes, lstm_out_channels)
    assert C.shape == (number_of_nodes, lstm_out_channels)

    H_tilde, H, C = layer(X, edge_index, edge_weight)

    assert H_tilde.shape == (number_of_nodes, lstm_out_channels)
    assert H.shape == (number_of_nodes, lstm_out_channels)
    assert C.shape == (number_of_nodes, lstm_out_channels)

    H_tilde, H, C = layer(X, edge_index, edge_weight, H, C)

    assert H_tilde.shape == (number_of_nodes, lstm_out_channels)
    assert H.shape == (number_of_nodes, lstm_out_channels)
    assert C.shape == (number_of_nodes, lstm_out_channels)


def test_evolve_gcn_h_layer():
    """
    Testing the Evolve GCN-H Layer.
    """
    number_of_nodes = 100
    edge_per_node = 10
    in_channels = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, edge_index = create_mock_data(number_of_nodes, edge_per_node, in_channels)
    X = X.to(device)
    edge_index = edge_index.to(device)
    edge_weight = create_mock_edge_weight(edge_index).to(device)

    layer = EvolveGCNH(in_channels=in_channels, num_of_nodes=number_of_nodes).to(device)

    X = layer(X, edge_index)

    assert X.shape == (number_of_nodes, in_channels)

    X = layer(X, edge_index, edge_weight)

    assert X.shape == (number_of_nodes, in_channels)


def test_evolve_gcn_o_layer():
    """
    Testing the Evolve GCN-O Layer.
    """
    number_of_nodes = 100
    edge_per_node = 10
    in_channels = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, edge_index = create_mock_data(number_of_nodes, edge_per_node, in_channels)
    X = X.to(device)
    edge_index = edge_index.to(device)
    edge_weight = create_mock_edge_weight(edge_index).to(device)

    layer = EvolveGCNO(in_channels=in_channels).to(device)

    X = layer(X, edge_index)

    assert X.shape == (number_of_nodes, in_channels)

    X = layer(X, edge_index, edge_weight)

    assert X.shape == (number_of_nodes, in_channels)
