from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.data.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.data.splitter import discrete_train_test_split


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        #self.recurrent_1 = DCRNN(node_features, 32, 2)
        self.linear = torch.nn.Linear(node_features, 1)

    def forward(self, x, edge_index, edge_weight):
        #h = self.recurrent_1(x, edge_index, edge_weight)
        #h = F.relu(h)
        h = self.linear(x)
        return h

dataset = ChickenpoxDatasetLoader().get_dataset()

train_dataset, test_dataset = discrete_train_test_split(dataset, train_ratio=0.8)

model = RecurrentGCN(node_features = 8)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

model.train()
loss = torch.nn.MSELoss()
for epoch in tqdm(range(200)):
    cost = 0
    for i, snapshot in enumerate(train_dataset):
        #print(snapshot.x)
        optimizer.zero_grad()
        out = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)     
        cost = cost + loss(snapshot.y, out)
        #print(i)
    cost = cost/((i+1))
    cost.backward()
    optimizer.step()

model.eval()
loss = 0
y, y_hat = [], []
for t, snapshot in enumerate(test_dataset):
    pred = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    y_hat.append(pred.detach().numpy())    
    y.append(snapshot.y.detach().numpy())  
y = np.concatenate(y)
y_hat = np.concatenate(y_hat)
print(y.shape)
print(r2_score(y, y_hat))

print(i,model.linear.bias)
print(i,model.linear.weight)
