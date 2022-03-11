import torch
import numpy as np
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric_temporal.nn.hetero import HeteroGCLSTM


def get_edge_array(n_count):
    return np.array([edge for edge in nx.gnp_random_graph(n_count, 0.1).edges()]).T


def create_hetero_mock_data(n_count, feature_dict):
    _x_dict = {'author': torch.FloatTensor(np.random.uniform(0, 1, (n_count, feature_dict['author']))),
               'paper': torch.FloatTensor(np.random.uniform(0, 1, (n_count, feature_dict['paper'])))}
    _edge_index_dict = {('author', 'writes', 'paper'): torch.LongTensor(get_edge_array(n_count))}

    data = HeteroData()
    data['author'].x = _x_dict['author']
    data['paper'].x = _x_dict['paper']
    data[('author', 'writes', 'paper')].edge_index = _edge_index_dict[('author', 'writes', 'paper')]
    data = T.ToUndirected()(data)

    return data.x_dict, data.edge_index_dict, data.metadata()


def test_hetero_gclstm_layer():
    """
        Testing the HeteroGCLSTM Layer.
    """
    number_of_nodes = 50
    feature_dict = {'author': 20, 'paper': 30}
    out_channels = 32

    x_dict, edge_index_dict, metadata = create_hetero_mock_data(number_of_nodes, feature_dict)

    layer = HeteroGCLSTM(in_channels_dict=feature_dict, out_channels=out_channels, metadata=metadata)

    h_dict, c_dict = layer(x_dict, edge_index_dict)

    assert h_dict['author'].shape == (number_of_nodes, out_channels)
    assert h_dict['paper'].shape == (number_of_nodes, out_channels)
    assert c_dict['author'].shape == (number_of_nodes, out_channels)
    assert c_dict['paper'].shape == (number_of_nodes, out_channels)

    h_dict, c_dict = layer(x_dict, edge_index_dict, h_dict, c_dict)

    assert h_dict['author'].shape == (number_of_nodes, out_channels)
    assert h_dict['paper'].shape == (number_of_nodes, out_channels)
    assert c_dict['author'].shape == (number_of_nodes, out_channels)
    assert c_dict['paper'].shape == (number_of_nodes, out_channels)
