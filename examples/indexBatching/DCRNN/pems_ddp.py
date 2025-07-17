import time 
import csv
import argparse
import uuid
import os

from torch_geometric_temporal.nn.recurrent import BatchedDCRNN as DCRNN
from torch_geometric_temporal.dataset import PemsBayDatasetLoader,PemsAllLADatasetLoader,PemsDatasetLoader
from utils import *

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from dask.distributed import LocalCluster
from dask.distributed import Client
from dask_pytorch_ddp import dispatch, results
from dask.distributed import wait as Wait


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Demo of DDP index-batching with PeMS-Bay, PeMS-All-LA, and PeMS")

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

    parser.add_argument(
     "--dask-cluster-file", type=str, default="", help="Dask scheduler file for the Dask CLI Interfance"
    )

    parser.add_argument(
     "-np","--npar", type=int, default=1, help="The number of GPUs/workers per node"
    )
    parser.add_argument(
     "--dataset", type=str, default="pems-bay", help="Which dataset is in use"
    )

    
    return parser.parse_args()


def train(args=None, epochs=None, batch_size=None, allGPU=False, debug=False, loader=None, start_time=None):

    worker_rank = int(dist.get_rank())
    gpu = worker_rank % 4   
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.set_device(device)
    
    world_size = dist.get_world_size()

    
    if allGPU == True:
        train_dataloader, val_dataloader, test_dataloader, edges, edge_weights, mean, std = loader.get_index_dataset(allGPU=gpu, batch_size=batch_size, world_size=world_size, ddp_rank=worker_rank) 
    else:
        train_dataloader, val_dataloader, test_dataloader, edges, edge_weights, mean, std = loader.get_index_dataset(batch_size=batch_size, world_size=world_size, ddp_rank=worker_rank) 
    

    
    # Move to GPU
    edge_index = edges.to(device)
    edge_weight = edge_weights.to(device)
    
    if allGPU == False:
        mean = mean.to(device)
        std = std.to(device)
        
    # Initialize model
    model = DCRNN(2, 2, K=3).to(device)
    if torch.cuda.is_available():
        model = DDP(model, gradient_as_bucket_view=True, device_ids=[device], output_device=[device])
    else:
        model = DDP(model, gradient_as_bucket_view=True)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # Training loop
    stats = []
    min_t = 9999
    min_v = 9999
    for epoch in range(epochs):
        train_dataloader.sampler.set_epoch(epoch)

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
            
            if debug and worker_rank == 0:
                print(f"Train Batch: {i}/{total}", end="\r")
                i+=1
            
        

        train_loss /= len(train_dataloader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        i = 0
        if debug and worker_rank == 0:
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
                
                if debug and worker_rank == 0:
                    print(f"Val Batch: {i}/{total}", end="\r")
                    i += 1

        # average valdiation across all ranks
        val_tensor = torch.tensor([val_loss, len(val_dataloader)])
        dist.reduce(val_tensor,dst=0, op=dist.ReduceOp.SUM)
        t2 = time.time()
        
        if worker_rank == 0:
            val_loss = val_tensor[0]/ val_tensor[1]
            
            # Print epoch metrics
            print(f"Epoch {epoch + 1}/{epochs}, Runtime: {t2 - t1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", flush=True)
            stats.append([epoch+1, t2 - t1, train_loss, float(val_loss)])

            min_t = min(min_t, train_loss)
            min_v = min(min_v, val_loss)
    if worker_rank == 0:
        print(f"Runtime: {round(t2 - start_time,2)}; Best Train MSE: {min_t}; Best Validation MSE: {min_v}", flush=True)

def main():
    args = parse_arguments()
    allGPU = args.gpu.lower() in ["true", "y", "t", "yes"]
    debug = args.debug.lower() in ["true", "y", "t", "yes"]
    batch_size = args.batch_size
    epochs = args.epochs
    npar = args.npar

    if args.dataset.lower() not in ["pems-bay","pemsallla", "pems"]:
        raise ValueError("Invalid argument for --dataset. --dataset must be 'pems-bay', 'pemsAllLA', or 'pems'")
  
    t1 = time.time() 
    
    # force the datasets to download before launching dask cluster
    if args.dataset.lower() == "pems-bay":
        loader = PemsBayDatasetLoader(index=True)
    if args.dataset.lower() == "pemsallla":
        loader = PemsAllLADatasetLoader(index=True)
    if args.dataset.lower() == "pems":
        loader = PemsDatasetLoader(index=True)
    
    
    if args.dask_cluster_file != "":
        client = Client(scheduler_file = args.dask_cluster_file)
    else:
        cluster = LocalCluster(n_workers=npar)
        client = Client(cluster)

    futures = dispatch.run(client, train,
                            args=args, debug=debug, epochs=epochs, batch_size=batch_size,allGPU=allGPU,loader=loader,
                            start_time=t1,
                            backend="gloo")
    
    key = uuid.uuid4().hex
    rh = results.DaskResultsHandler(key)
    rh.process_results(".", futures, raise_errors=False)
    client.shutdown()


if __name__ == "__main__":
    main()