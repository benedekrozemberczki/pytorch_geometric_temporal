import torch.optim as optim
import torch.nn as nn
import torch
import math
import networkx as nx
import numpy as np
from torch_geometric_temporal.nn import GMAN

def create_mock_data(number_of_nodes, edge_per_node, in_channels):
    """
    Creating a mock feature matrix and edge index.
    """
    graph = nx.watts_strogatz_graph(number_of_nodes, edge_per_node, 0.5)
    edge_index = torch.LongTensor(np.array([edge for edge in graph.edges()]).T)
    X = torch.FloatTensor(np.random.uniform(-1, 1, (number_of_nodes, in_channels)))
    return X, edge_index

time_slot = 5
T = 24 * 60 // time_slot  # Number of time steps in one day
L = 1
K = 8
d = 8
epochs = 3
# generate data
num_his = 12
num_pred = 12
num_nodes = 50
num_sample = 100
learning_rate = 0.001
decay_epoch = 10
batch_size = 32
trainX = torch.rand(num_sample,num_his, num_nodes)
trainY = torch.rand(num_sample,num_pred, num_nodes)
mean, std = torch.mean(trainX), torch.std(trainX)
SE, _ = create_mock_data(number_of_nodes=num_nodes, edge_per_node=8, in_channels=64)
trainTE = torch.zeros((num_sample, num_his + num_pred, 2))
for i in range(num_his+num_pred):
    x, _ = create_mock_data(number_of_nodes=num_sample, edge_per_node=8, in_channels=2)
    trainTE[:,i,:] = x
# build model
model = GMAN(SE, L, K, d, num_his, bn_decay=0.1)
loss_criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=decay_epoch,
                                      gamma=0.9)

# start training
num_train, _, num_vertex = trainX.shape
train_num_batch = math.ceil(num_train / batch_size)
# shuffle
permutation = torch.randperm(num_train)
trainX = trainX[permutation]
trainTE = trainTE[permutation]
trainY = trainY[permutation]
for i in range(epochs):
    for batch_idx in range(train_num_batch):
        start_idx = batch_idx * batch_size
        end_idx = min(num_train, (batch_idx + 1) * batch_size)
        X = trainX[start_idx: end_idx]
        TE = trainTE[start_idx: end_idx]
        label = trainY[start_idx: end_idx]
        optimizer.zero_grad()
        pred = model(X, TE)
        pred = pred * std + mean
        loss_batch = loss_criterion(pred, label)
        loss_batch.backward()
        optimizer.step()
        