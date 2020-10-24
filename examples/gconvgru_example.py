import torch
import random
import numpy as np
import networkx as nx
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU

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


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(RecurrentGCN, self).__init__()
        self.recurrent_1 = GConvGRU(node_features, 32, 5)
        self.recurrent_2 = GConvGRU(32, 16, 5)
        self.linear = torch.nn.Linear(16, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.recurrent_1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.recurrent_2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


node_features = 100
node_count = 1000
num_classes = 10
edge_per_node = 15
epochs = 200
learning_rate = 0.01
weight_decay = 5e-4

model = RecurrentGCN(node_features=node_features, num_classes=num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

model.train()

for epoch in range(epochs):
    optimizer.zero_grad()
    x, edge_index = create_mock_data(node_count, edge_per_node, node_features)
    edge_weight = create_mock_edge_weight(edge_index)
    scores = model(x, edge_index, edge_weight)
    target = create_mock_target(node_count, num_classes)
    loss = F.nll_loss(scores, target)
    loss.backward()
    optimizer.step()