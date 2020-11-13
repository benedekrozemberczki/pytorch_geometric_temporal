from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.data.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.data.splitter import discrete_train_test_split


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent_1 = DCRNN(node_features, 32, 2)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent_1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

dataset = ChickenpoxDatasetLoader().get_dataset()

train_dataset, test_dataset = discrete_train_test_split(dataset, train_ratio=0.8)

model = RecurrentGCN(node_features = 21)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in tqdm(range(20)):
    x = 0
    for i, snapshot in enumerate(train_dataset):
        optimizer.zero_grad()
        out = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)     
        loss = torch.mean(torch.abs(out-snapshot.y))   
        x = x + loss.item()
        loss.backward()
        optimizer.step()
    print(x/(i+1))

model.eval()
loss = 0
for t, snapshot in enumerate(train_dataset):
    out = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)     
    loss = loss + torch.mean(torch.abs(out-snapshot.y)).item()  
print(loss/(t+1))
