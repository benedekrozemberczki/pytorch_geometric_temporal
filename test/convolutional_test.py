import torch
import numpy as np
import networkx as nx
from torch_geometric_temporal.nn.convolutional import TemporalConv, STConv

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
    return torch.LongTensor([random.randint(0, number_of_classes-1) for node in range(number_of_nodes)])

def create_mock_sequence(sequence_length, number_of_nodes, edge_per_node, in_channels, number_of_classes):
    """
    Creating mock sequence data
    
    Note that this is a static graph discrete signal type sequence
    The target is the "next" iterm in the sequence
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
