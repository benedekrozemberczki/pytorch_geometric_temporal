import torch
import random
import numpy as np
import networkx as nx
from torch import nn
from torch_geometric_temporal.nn.convolutional import ASTGCN

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
epochs = 2
learning_rate = 0.01
weight_decay = 5e-4


num_for_predict = 12
len_input = 12
nb_time_strides = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
node_features = 2
nb_block = 2
K = 3
nb_chev_filter = 64
nb_time_filter = 64
batch_size = 32

x, edge_index = create_mock_data(node_count, edge_per_node, node_features)
model = ASTGCN(device, nb_block, node_features, K, nb_chev_filter, nb_time_filter, nb_time_strides, num_for_predict, len_input, node_count)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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
criterion = torch.nn.MSELoss().to(device)
for epoch in range(epochs):
    for batch_data in train_loader:
        encoder_inputs, labels = batch_data
        optimizer.zero_grad()
        outputs = model(encoder_inputs, edge_index)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()