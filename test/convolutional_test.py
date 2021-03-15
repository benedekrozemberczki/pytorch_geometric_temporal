import torch
import numpy as np
import networkx as nx
from torch_geometric_temporal.nn.convolutional import TemporalConv, STConv, ASTGCN
from torch_geometric.utils import to_scipy_sparse_matrix

def create_mock_data(number_of_nodes, edge_per_node, in_channels):
    """
    Creating a mock feature matrix and edge index.
    """
    graph = nx.watts_strogatz_graph(number_of_nodes, edge_per_node, 0.5)
    edge_index = torch.LongTensor(np.array([edge for edge in graph.edges()]).T)
    X = torch.FloatTensor(np.random.uniform(-1, 1, (number_of_nodes, in_channels)))
    return X, edge_index

def create_mock_edge_weight(edge_index):
    """
    Creating a mock edge weight tensor.
    """
    return torch.FloatTensor(np.random.uniform(0, 1, (edge_index.shape[1])))

def create_mock_target(number_of_nodes, number_of_classes):
    """
    Creating a mock target vector.
    """
    return torch.LongTensor([np.random.randint(0, number_of_classes-1) for node in range(number_of_nodes)])

def create_mock_sequence(sequence_length, number_of_nodes, edge_per_node, in_channels, number_of_classes):
    """
    Creating mock sequence data
    
    Note that this is a static graph discrete signal type sequence
    The target is the "next" item in the sequence
    """

    input_sequence = torch.zeros(sequence_length, number_of_nodes, in_channels)
    
    X, edge_index = create_mock_data(number_of_nodes=number_of_nodes, edge_per_node=edge_per_node, in_channels=in_channels)
    edge_weight = create_mock_edge_weight(edge_index)
    targets = create_mock_target(number_of_nodes, number_of_classes)

    for t in range(sequence_length):
        input_sequence[t] = X+t

    return input_sequence, targets, edge_index, edge_weight

def create_mock_batch(batch_size, sequence_length, number_of_nodes, edge_per_node, in_channels, number_of_classes):
    """
    Creating a mock batch of sequences
    """
    batch = torch.zeros(batch_size, sequence_length, number_of_nodes, in_channels)
    batch_targets = torch.zeros(batch_size, number_of_nodes, dtype=torch.long)
    
    for b in range(batch_size):
        input_sequence, targets, edge_index, edge_weight = create_mock_sequence(sequence_length, number_of_nodes, edge_per_node, in_channels, number_of_classes)
        batch[b] = input_sequence
        batch_targets[b] = targets

    return batch, batch_targets, edge_index, edge_weight

def test_temporalconv():
    """
    Testing the temporal block in STGCN
    """
    batch_size = 10
    sequence_length = 5

    number_of_nodes = 300
    in_channels = 100
    edge_per_node = 15
    out_channels = 10
    batch, batch_targets, edge_index, edge_weight = create_mock_batch(batch_size, sequence_length, number_of_nodes, edge_per_node, in_channels, out_channels)

    kernel_size = 3
    temporal_conv = TemporalConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    H = temporal_conv(batch)
    assert H.shape == (batch_size, sequence_length-(kernel_size-1), number_of_nodes, out_channels)    

def test_stconv():
    """
    Testing STConv block in STGCN
    """
    batch_size = 10
    sequence_length = 5

    number_of_nodes = 300
    in_channels = 100
    edge_per_node = 15
    out_channels = 10
    batch, batch_targets, edge_index, edge_weight = create_mock_batch(batch_size, sequence_length, number_of_nodes, edge_per_node, in_channels, out_channels)

    kernel_size = 3
    stconv = STConv(num_nodes=number_of_nodes, in_channels=in_channels, hidden_channels=8, out_channels=out_channels, kernel_size=3, K=2)
    H = stconv(batch, edge_index, edge_weight)
    assert H.shape == (batch_size, sequence_length-2*(kernel_size-1), number_of_nodes, out_channels)

def test_astgcn():
    """
    Testing ASTGCN block
    """
    node_count = 307
    num_classes = 10
    edge_per_node = 15


    num_of_vertices = node_count # 307
    num_for_predict = 12
    len_input = 12
    nb_time_strides = 1

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_features = 1
    nb_block = 2
    K = 3
    nb_chev_filter = 64
    nb_time_filter = 64
    batch_size = 32

    x, edge_index = create_mock_data(node_count, edge_per_node, node_features)
    adj_mx = to_scipy_sparse_matrix(edge_index)
    model = ASTGCN(DEVICE, nb_block, node_features, K, nb_chev_filter, nb_time_filter, nb_time_strides, adj_mx.toarray(),num_for_predict, len_input, node_count)

    T = len_input
    x_seq = torch.zeros([batch_size,node_count, node_features,T]).to(DEVICE)
    target_seq = torch.zeros([batch_size,node_count,T]).to(DEVICE)
    for b in range(batch_size):
        for t in range(T):
            x, edge_index = create_mock_data(node_count, edge_per_node, node_features)
            x_seq[b,:,:,t] = x
            target = create_mock_target(node_count, num_classes)
            target_seq[b,:,t] = target
    shuffle = True
    train_dataset = torch.utils.data.TensorDataset(x_seq, target_seq)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    criterion = torch.nn.MSELoss().to(DEVICE)
    for batch_data in train_loader:
        encoder_inputs, labels = batch_data
        outputs = model(encoder_inputs)
    assert outputs.shape == (batch_size, node_count, num_for_predict)
