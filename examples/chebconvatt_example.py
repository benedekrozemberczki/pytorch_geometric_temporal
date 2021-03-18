import torch
import random
import numpy as np
import networkx as nx
from torch_geometric_temporal.nn import ChebConvAtt
from torch_geometric.transforms import LaplacianLambdaMax
from torch_geometric.data import Data

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

node_count = 307
num_classes = 10
edge_per_node = 15


len_input = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
node_features = 2
K = 3
nb_chev_filter = 64
batch_size = 32

x, edge_index = create_mock_data(node_count, edge_per_node, node_features)
model = ChebConvAtt(node_features, nb_chev_filter, K)
spatial_attention = torch.rand(batch_size,node_count,node_count)
spatial_attention = torch.nn.functional.softmax(spatial_attention, dim=1)
model.train()
T = len_input
x_seq = torch.zeros([batch_size,node_count, node_features,T]).to(device)
target_seq = torch.zeros([batch_size,node_count,T]).to(device)
for b in range(batch_size):
    for t in range(T):
        x, edge_index = create_mock_data(node_count, edge_per_node, node_features)
        x_seq[b,:,:,t] = x
        target = create_mock_target(node_count, num_classes)
        target_seq[b,:,t] = target
shuffle = True
train_dataset = torch.utils.data.TensorDataset(x_seq, target_seq)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
for batch_data in train_loader:
    encoder_inputs, labels = batch_data
    data = Data(edge_index=edge_index, edge_attr=None, num_nodes=node_count)
    lambda_max = LaplacianLambdaMax()(data).lambda_max
    outputs = []
    for time_step in range(T):
        outputs.append(torch.unsqueeze(model(encoder_inputs[:,:,:,time_step], edge_index, spatial_attention, lambda_max = lambda_max), -1))
    spatial_gcn = torch.nn.functional.relu(torch.cat(outputs, dim=-1)) # (b,N,F,T) # (b,N,F,T)