import numpy as np
import time
import csv
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import A3TGCN2
from torch_geometric_temporal.dataset import PemsDatasetLoader
import argparse
from utils import *


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

# Making the model 
class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, batch_size):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features,  out_channels=32, periods=periods,batch_size=batch_size) # node_features=2, periods=12
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index) # x [b, 207, 2, 12]  returns h [b, 207, 12]
        h = F.relu(h) 
        h = self.linear(h)
        return h



def train(train_dataloader, val_dataloader, batch_size, epochs, edges, DEVICE, allGPU=False, debug=False):
    
    # Create model and optimizers
    model = TemporalGNN(node_features=2, periods=12, batch_size=batch_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    
    stats = []
    t_mse = []
    v_mse = []

    
    edges = edges.to(DEVICE)
    for epoch in range(epochs):
        step = 0
        loss_list = []
        t1 = time.time()
        i = 1
        total = len(train_dataloader)
        
        for batch in train_dataloader:
            X_batch, y_batch = batch
            
            # Need to permute based on expected input shape for ATGCN
            if allGPU:
                X_batch = X_batch.permute(0, 2, 3, 1)
                y_batch = y_batch[...,0].permute(0, 2, 1)
            else:
                X_batch = X_batch.permute(0, 2, 3, 1).to(DEVICE).float()
                y_batch = y_batch[...,0].permute(0, 2, 1).to(DEVICE).float()
            

            y_hat = model(X_batch, edges)         # Get model predictions
            loss = loss_fn(y_hat, y_batch) # Mean squared error #loss = torch.mean((y_hat-labels)**2)  sqrt to change it to rmse
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step= step+ 1
            loss_list.append(loss.item())

            if debug:
                print(f"Train Batch: {i}/{total}", end="\r")
                i+=1


        model.eval()
        step = 0
        # Store for analysis
        total_loss = []
        i = 1
        total = len(val_dataloader)
        if debug:
            print("                      ", end="\r")
        with torch.no_grad():
            for batch in val_dataloader:
                X_batch, y_batch = batch

                # Need to permute based on expected input shape for ATGCN
                if allGPU:
                    X_batch = X_batch.permute(0, 2, 3, 1)
                    y_batch = y_batch[...,0].permute(0, 2, 1)
                else:
                    X_batch = X_batch.permute(0, 2, 3, 1).to(DEVICE).float()
                    y_batch = y_batch[...,0].permute(0, 2, 1).to(DEVICE).float()
                
                # Get model predictions
                y_hat = model(X_batch, edges)
                # Mean squared error
                loss = loss_fn(y_hat, y_batch)
                total_loss.append(loss.item())
                
                if debug:
                    print(f"Val Batch: {i}/{total}", end="\r")
                    i += 1
                
        t2 = time.time()
        print("Epoch {} time: {:.4f} train MSE: {:.4f} Test MSE: {:.4f}".format(epoch,t2 - t1, sum(loss_list)/len(loss_list), sum(total_loss)/len(total_loss)))
        stats.append([epoch, t2-t1, sum(loss_list)/len(loss_list), sum(total_loss)/len(total_loss)])
        t_mse.append(sum(loss_list)/len(loss_list))
        v_mse.append(sum(total_loss)/len(total_loss))
    return min(t_mse), min(v_mse)
        

  





def main():
    """
    Note that error (MSE) is calculated over the standardized data values to mimic the existing A3T-GCN2 workflow. 
    For comparision to existing work, it is better to reverse the standardization proccess
    and use MAE as error.
    """
    args = parse_arguments()
    allGPU = args.gpu.lower() in ["true", "y", "t", "yes"]
    debug = args.debug.lower() in ["true", "y", "t", "yes"]
    batch_size = args.batch_size
    epochs = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shuffle= True


    start = time.time()
    p1 = time.time() 
    indexLoader = PemsDatasetLoader(index=True)
    if allGPU:
        train_dataloader, val_dataloader, test_dataloader, edges, edge_weights, mean, std = indexLoader.get_index_dataset(batch_size=batch_size, shuffle=shuffle, allGPU=0) 
    else:
        train_dataloader, val_dataloader, test_dataloader, edges, edge_weights, mean, std = indexLoader.get_index_dataset(batch_size=batch_size, shuffle=shuffle) 
    p2 = time.time() 
    t_mse, v_mse = train(train_dataloader, val_dataloader, batch_size, epochs, edges, device, debug=debug)
    end = time.time()

    print(f"Runtime: {round(end - start,2)}; T-MSE: {round(t_mse, 3)}; V-MSE: {round(v_mse, 3)}")

if __name__ == "__main__":
    main()