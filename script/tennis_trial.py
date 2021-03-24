from torch_geometric_temporal.dataset import *
from torch_geometric_temporal.signal import temporal_signal_split
import numpy as np
import sys

event_id = sys.argv[1]
N = int(sys.argv[2])
offset = int(sys.argv[3])
num_epochs = int(sys.argv[4])
if len(sys.argv) > 5:
    mode = sys.argv[5]
else:
    mode = None
print(mode)

loader = TwitterTennisDatasetLoader(event_id, N, mode, offset)
dataset = loader.get_dataset()

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.5)

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
from tqdm import tqdm

if mode == "encoded":
    model = RecurrentGCN(node_features = 16)
elif mode == "diagonal":
    model = RecurrentGCN(node_features = N)
else:
    model = RecurrentGCN(node_features = 2)
    
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()

train_mse = []
for epoch in tqdm(range(num_epochs)):
    cost = 0
    epoch_ndcg = []
    for time, snapshot in enumerate(train_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
    cost = cost / (time+1)
    train_mse.append(cost.detach().numpy())
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()
#print(train_mse)
print("train mean mse:", np.mean(train_mse))

test_mse = []
cost = 0
for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    new_cost = torch.mean((y_hat-snapshot.y)**2)
    cost = cost + new_cost
    test_mse.append(new_cost.detach().numpy())
cost = cost / (time+1)
print("test mse:", cost.detach().numpy())
#print(list(zip(range(60,120),test_mse)))
print("done")