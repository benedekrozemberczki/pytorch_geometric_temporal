import time 
import csv
import argparse
import uuid
import os

import numpy as np
import time
import csv
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import A3TGCN2

from torch_geometric_temporal.dataset import PemsBayDatasetLoader,PemsAllLADatasetLoader,PemsDatasetLoader,METRLADatasetLoader


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
    

    
    model = TemporalGNN(node_features=2, periods=12, batch_size=batch_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    
    stats = []
    t_mse = []
    v_mse = []

    if torch.cuda.is_available():
        model = DDP(model, gradient_as_bucket_view=True, device_ids=[device], output_device=[device])
    else:
        model = DDP(model, gradient_as_bucket_view=True)



    # Training loop
    stats = []
    min_t = 9999
    min_v = 9999

    edges = edges.to(device)
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
                X_batch = X_batch.permute(0, 2, 3, 1).to(device).float()
                y_batch = y_batch[...,0].permute(0, 2, 1).to(device).float()
            


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
        val_loss = 0
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
                    X_batch = X_batch.permute(0, 2, 3, 1).to(device).float()
                    y_batch = y_batch[...,0].permute(0, 2, 1).to(device).float()
                
                # Get model predictions
                y_hat = model(X_batch, edges)
                # Mean squared error
                loss = loss_fn(y_hat, y_batch)
                val_loss += loss.item()
                
                if debug:
                    print(f"Val Batch: {i}/{total}", end="\r")
                    i += 1
        
        val_tensor = torch.tensor([val_loss, len(val_dataloader)])
        dist.reduce(val_tensor,dst=0, op=dist.ReduceOp.SUM)
        t2 = time.time()
        
        if worker_rank == 0:
            val_loss = val_tensor[0]/ val_tensor[1]
            
            t2 = time.time()
            print("Epoch {} time: {:.4f} Train MSE: {:.4f} Validation MSE: {:.4f}".format(epoch,t2 - t1, sum(loss_list)/len(loss_list), val_loss.item()))
            stats.append([epoch, t2-t1, sum(loss_list)/len(loss_list), val_loss.item()])
            t_mse.append(sum(loss_list)/len(loss_list))
            v_mse.append(val_loss.item())



    print(f"Runtime: {round(t2 - start_time,2)}; Best Train MSE: {min(t_mse)}; Best Validation MSE: {v_mse}", flush=True)

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
    npar = args.npar

    if args.dataset.lower() not in ["pems-bay","pemsallla", "pems", "metr-la"]:
        raise ValueError("Invalid argument for --dataset. --dataset must be 'metr-la', 'pems-bay', 'pemsAllLA', or 'pems'")
  
    t1 = time.time() 
    
    # force the datasets to download before launching dask cluster
    if args.dataset.lower() == "pems-bay":
        loader = PemsBayDatasetLoader(index=True)
    if args.dataset.lower() == "pemsallla":
        loader = PemsAllLADatasetLoader(index=True)
    if args.dataset.lower() == "pems":
        loader = PemsDatasetLoader(index=True)
    if args.dataset.lower() == "metr-la":
        loader = METRLADatasetLoader(index=True)
    
    
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