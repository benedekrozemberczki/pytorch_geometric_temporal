import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.nn.recurrent import TGCN2
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time

# --- Hyperparams ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
WINDOW = 12
IN_CHANNELS = 2
OUT_CHANNELS = 1
EPOCHS = 100

# --- Load dataset ---
loader = METRLADatasetLoader()
raw_dataset = loader.get_dataset(num_timesteps_in=WINDOW, num_timesteps_out=WINDOW)
train_dataset, test_dataset = temporal_signal_split(raw_dataset, train_ratio=0.8)


train_x = np.array(train_dataset.features) # (27399, 207, 2, 12)
train_y = np.array(train_dataset.targets) # (27399, 207, 12)
test_x = np.array(test_dataset.features) # (6850, 207, 2, 12)
test_y = np.array(test_dataset.targets) # (6850, 207, 12)



train_loader = DataLoader(TensorDataset(torch.tensor(train_x).to(DEVICE), torch.tensor(train_y).to(DEVICE)),
                          batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

test_loader = DataLoader(TensorDataset(torch.tensor(test_x).to(DEVICE), torch.tensor(test_y).to(DEVICE)),
                          batch_size=BATCH_SIZE, shuffle=False, drop_last=True)


# --- Use static graph from first snapshot ---
edge_index = train_dataset[0].edge_index.long().to(DEVICE)
edge_weight = train_dataset[0].edge_attr.float().to(DEVICE)

# --- Model ---
class BatchedTGCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super().__init__()
        self.tgnn = TGCN2(in_channels, hidden_dim, 1)
        self.linear = nn.Linear(hidden_dim, out_channels)

    def forward(self, x, edge_index, edge_weight):
        # x: [B, N, F, T]
        B, N, Fin, T = x.shape
        
        h = None
        output_sequence = []
        for t in range(T):
            h = self.tgnn(x[..., t], edge_index, edge_weight, h)  # h: [B, N, hidden_dim]
            h_t = F.relu(h)
            out_t = self.linear(h_t).unsqueeze(1)  # [B, N, output_dim] → [B, 1, N, output_dim]
            output_sequence.append(out_t)

        return torch.cat(output_sequence, dim=1) # [B, T, N, output_dim]

model = BatchedTGCN(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, hidden_dim=32, ).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()


for epoch in range(EPOCHS):
    model.train()
    epoch_loss = []
    i = 1
    total = len(train_loader)
    t1 = time.time()
    for x, y in train_loader:
        y_hat = model(x, edge_index, edge_weight)
        y_hat = y_hat.squeeze()
        y = y.permute(0, 2, 1)  # [B, T, N]

        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss.append(loss.item())
        
        print(f"Train Batch: {i}/{total}", end="\r")
        i+=1
    
    
    print("                      ", end="\r")
    model.eval()
    test_loss = []
    i = 1
    total = len(test_loader)
    with torch.no_grad():
        for x, y in test_loader:
            y_hat = model(x, edge_index, edge_weight)
            y_hat = y_hat.squeeze()
            y = y.permute(0, 2, 1)  # [B, T, N]
            
            loss = loss_fn(y_hat, y)
            test_loss.append(loss.item())
            
            
            print(f"Test Batch: {i}/{total}", end="\r")
            i+=1

    t2 = time.time()
    
    """
    Note that MSE is calculated over the standardized data values. 
    For comparision to existing work, it is better to reverse the standardization proccess
    and use MAE as error.
    """
    print("Epoch {} time: {:.4f} train MSE: {:.4f} Test MSE: {:.4f}".format(epoch,t2 - t1,np.mean(epoch_loss), np.mean(test_loss)))
