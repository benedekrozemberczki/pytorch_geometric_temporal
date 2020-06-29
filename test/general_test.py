import torch
import numpy as np
import networkx as nx
from torch_geometric_temporal.nn.conv import GConvLSTM



def test_gconv_lstm_layer():
    """
    Testing the GConvLSTM Layer.
    """

    graph = nx.watts_strogatz_graph(100, 10, 0.5)
    features = torch.FloatTensor(np.random.uniform(-1, 1, (100, 64)))

    layer = GConvLSTM(in_channels=64, out_channels=16, K=2, number_of_nodes=100)

    assert layer.in_channels == 64
    assert layer.out_channels == 16
