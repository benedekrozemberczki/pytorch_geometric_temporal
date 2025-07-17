from torch_geometric_temporal.nn.recurrent import BatchedDCRNN as DCRNN
from torch_geometric_temporal.dataset import PemsDatasetLoader
import torch.optim as optim

import argparse
import csv
import os
import time 
from utils import *


def parse_arguments():
    parser = argparse.ArgumentParser(description="Demo of index batching with Pems dataset")

    parser.add_argument(
        "-e", "--epochs", type=int, default=30, help="The desired number of training epochs"
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




def train(train_dataloader, val_dataloader, mean, std, edges, edge_weights, epochs, seq_length, num_nodes, num_features, allGPU=False, debug=False):
    
    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = edges.to(device)
    edge_weight = edge_weights.to(device)
    
    if allGPU == False:
        mean = mean.to(device)
        std = std.to(device)

    # Initialize model
    model = DCRNN(num_features, num_features, K=3).to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    stats = []
    min_t = 9999
    min_v = 9999
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        i = 1
        total = len(train_dataloader)
        t1 = time.time()
        for batch in train_dataloader:
            X_batch, y_batch = batch

            if allGPU == False:
                X_batch = X_batch.to(device).float()
                y_batch = y_batch.to(device).float()

            # Forward pass
            outputs = model(X_batch, edge_index, edge_weight)  # Shape: (batch_size, seq_length, num_nodes, out_channels)

            # Calculate loss
            loss = masked_mae_loss((outputs * std) + mean, (y_batch * std) + mean)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if debug:
                print(f"Train Batch: {i}/{total}", end="\r")
                i+=1
            # break
        

        train_loss /= len(train_dataloader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        i = 0
        if debug:
            print("                      ", end="\r")
        total = len(val_dataloader)
        with torch.no_grad():
            for batch in val_dataloader:
                X_batch, y_batch = batch

                if allGPU == False:
                    X_batch = X_batch.to(device).float()
                    y_batch = y_batch.to(device).float()

                # Forward pass
                outputs = model(X_batch, edge_index, edge_weight)

                # Calculate loss
                loss = masked_mae_loss((outputs * std) + mean, (y_batch * std) + mean)
                val_loss += loss.item()
                if debug:
                    print(f"Val Batch: {i}/{total}", end="\r")
                    i += 1
                

        val_loss /= len(val_dataloader)
        t2 = time.time()
        # Print epoch metrics
        print(f"Epoch {epoch + 1}/{epochs}, Runtime: {t2 - t1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", flush=True)
        stats.append([epoch+1, t2 - t1, train_loss, val_loss])

        min_t = min(min_t, train_loss)
        min_v = min(min_v, val_loss)
    
    return min_t, min_v

def main():
  


    args = parse_arguments()
    allGPU = args.gpu.lower() in ["true", "y", "t", "yes"]
    debug = args.debug.lower() in ["true", "y", "t", "yes"]
    batch_size = args.batch_size
    epochs = args.epochs

    t1 = time.time()
    loader = PemsDatasetLoader(index=True)
    if allGPU:
        train_dataloader, val_dataloader, test_dataloader, edges, edge_weights, means, stds = loader.get_index_dataset(allGPU=0, batch_size=batch_size)
    else:
        train_dataloader, val_dataloader, test_dataloader, edges, edge_weights, means, stds = loader.get_index_dataset(batch_size=batch_size)
    
       
    t_min, v_min = train(train_dataloader, val_dataloader, means, stds, edges, edge_weights, epochs, 12,11160,2, allGPU=allGPU, debug=debug)
    t2 = time.time()
    print(f"Runtime: {round(t2 - t1,2)}; Best Train MSE: {t_min}; Best Validation MSE: {v_min}")
    
if __name__ == "__main__":
    main()