import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric_temporal.dataset import PemsAllLADatasetLoader
from torch_geometric_temporal.nn.recurrent import TGCN2
import argparse
import time


def parse_arguments():
    parser = argparse.ArgumentParser(description="Demo of index batching with PemsBay dataset")

    parser.add_argument(
        "-e", "--epochs", type=int, default=100, help="The desired number of training epochs"
    )
    parser.add_argument(
        "-bs", "--batch-size", type=int, default=64, help="The desired batch size"
    )
    parser.add_argument(
        "-g", "--gpu", type=str, default="False", help="Should data be preprocessed and migrated directly to the GPU"
    )
    parser.add_argument(
        "-d", "--debug", type=str, default="False", help="Print values for debugging"
    )
    return parser.parse_args()

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

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean() 

def train(train_dataloader, val_dataloader, mean, std, batch_size, epochs, edge_index, edge_weight, DEVICE, allGPU=False, debug=False):
    # currently predicting speed and time of day. This can be changed to just predict speed.
    model = BatchedTGCN(in_channels=2, out_channels=2, hidden_dim=32, ).to(DEVICE)

    edge_index = edge_index.to(DEVICE)
    edge_weight = edge_weight.to(DEVICE)
    
    if not allGPU:
        mean = mean.to(DEVICE)
        std = std.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    t_maes = []
    v_maes = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = []
        i = 1
        total = len(train_dataloader)
        t1 = time.time()
        for x, y in train_dataloader:
            
            if allGPU:
                x = x.permute(0,2,3,1).float()
                y = y.float()

            else:
                x = x.permute(0,2,3,1).to(DEVICE).float()
                y = y.to(DEVICE).float()
        
            y_hat = model(x, edge_index, edge_weight).squeeze()            
            loss = masked_mae_loss((y_hat * std) + mean, (y * std) + mean)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss.append(loss.item())
            if debug:
                    print(f"Train Batch: {i}/{total}", end="\r")
                    i+=1
        
        if debug:
                print("                      ", end="\r")
        model.eval()
        test_loss = []
        i = 1
        total = len(val_dataloader)
        with torch.no_grad():
            for x, y in val_dataloader:
                if allGPU:
                    x = x.permute(0,2,3,1).float()
                    y = y.float()
                else:
                    x = x.permute(0,2,3,1).to(DEVICE).float()
                    y = y.to(DEVICE).float()

                y_hat = model(x, edge_index, edge_weight).squeeze()

                loss = masked_mae_loss((y_hat * std) + mean, (y * std) + mean)
                test_loss.append(loss.item())
               
                if debug:
                    print(f"Test Batch: {i}/{total}", end="\r")
                    i+=1

       
        t_maes.append(np.mean(epoch_loss))  
        v_maes.append(np.mean(test_loss))  
        t2 = time.time()
 
        print(f"Epoch {epoch + 1}/{epochs}, Runtime: {t2 - t1}, Train Loss: {np.mean(epoch_loss):.4f}, Val Loss: {np.mean(test_loss):.4f}", flush=True)

    return min(t_maes).item(), min(v_maes).item()

def main():
    args = parse_arguments()
    allGPU = args.gpu.lower() in ["true", "y", "t", "yes"]
    debug = args.debug.lower() in ["true", "y", "t", "yes"]
    batch_size = args.batch_size
    epochs = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shuffle= True

    start = time.time()
    p1 = time.time() 
    indexLoader = PemsAllLADatasetLoader(index=True)
    if allGPU:
        train_dataloader, val_dataloader, test_dataloader, edges, edge_weights, mean, std = indexLoader.get_index_dataset(batch_size=batch_size, shuffle=shuffle, allGPU=0) 
    else:
        train_dataloader, val_dataloader, test_dataloader, edges, edge_weights, mean, std = indexLoader.get_index_dataset(batch_size=batch_size, shuffle=shuffle) 
    p2 = time.time() 
    t_mse, v_mse = train(train_dataloader, val_dataloader, mean, std, batch_size, epochs, edges, edge_weights, device, debug=debug)
    end = time.time()

    print(f"Runtime: {round(end - start,2)}; T-MAE: {round(t_mse, 5)}; V-MAE: {round(v_mse, 5)}")

if __name__ == "__main__":
    main()