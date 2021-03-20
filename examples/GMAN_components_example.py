import torch.nn as nn
import torch
import math
import networkx as nx
import numpy as np
from torch_geometric_temporal.nn import STEmbedding, STAttBlock, transformAttention

def create_mock_data(number_of_nodes, edge_per_node, in_channels):
    """
    Creating a mock feature matrix and edge index.
    """
    graph = nx.watts_strogatz_graph(number_of_nodes, edge_per_node, 0.5)
    edge_index = torch.LongTensor(np.array([edge for edge in graph.edges()]).T)
    X = torch.FloatTensor(np.random.uniform(-1, 1, (number_of_nodes, in_channels)))
    return X, edge_index
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 8
d = 8
D = K * d
bn_decay = 0.1
num_his = 12
num_pred = 12
num_nodes = 50
num_sample = 100
batch_size = 32
SE, _ = create_mock_data(number_of_nodes=num_nodes, edge_per_node=8, in_channels=64)
trainTE = torch.zeros((num_sample, num_his + num_pred, 2))
for i in range(num_his+num_pred):
    x, _ = create_mock_data(number_of_nodes=num_sample, edge_per_node=8, in_channels=2)
    trainTE[:,i,:] = x
# layers
STEmbedding_layer = STEmbedding(D, bn_decay).to(device)
STAttBlock_layer = STAttBlock(K, d, bn_decay).to(device)
transformAttention_layer = transformAttention(K, d, bn_decay).to(device)
TE = trainTE[:batch_size]
# STE
STE = STEmbedding_layer(SE, TE)
STE_his = STE[:, :num_his]
STE_pred = STE[:, num_his:]
print(STE.shape) # (batch_size, num_his+num_pred, num_nodes,D)
X = torch.rand(batch_size,num_his,num_nodes,D).to(device)
# STAtt
X = STAttBlock_layer(X, STE_his)
print(X.shape) # (batch_size, num_his, num_nodes,D)
# transAtt
X = transformAttention_layer(X, STE_his, STE_pred)
print(X.shape) # (batch_size, num_his, num_nodes,D)
        