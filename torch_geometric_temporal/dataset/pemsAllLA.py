import os
import ssl
import requests
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from ..signal import StaticGraphTemporalSignal
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import pickle
from typing import Tuple

class PemsAllLADatasetLoader(object):
    """
    A traffic forecasting dataset that covers Los Angeles. This traffic dataset is collected by California 
    Transportation Agencies (CalTrans) Performance Measurement System (PeMS). 

    For details see: `"Graph-partitioning-based diffusion convolutional 
    recurrent neural network for large-scale traffic forecasting" 
    <https://arxiv.org/abs/1909.11197>`_

    Args:
        raw_data_dir (string, optional): The directory to download the PeMS-All-LA files to. 
            Defaults to "data/".
        index (bool, optional): If True, initializes the dataloader to use index-based batching.
            Defaults to False.
    """

    def __init__(self, raw_data_dir=os.path.join(os.getcwd(), "data"),index=False):
        
        if not index:
            NotImplementedError("The PeMSAllLA dataset does not support batching without the index-method")
        else:
            from ..signal.index_dataset import IndexDataset
            import pandas as pd

            self.pd = pd
            self.IndexDataset = IndexDataset

        super(PemsAllLADatasetLoader, self).__init__()
        self.index = index
        self.raw_data_dir = raw_data_dir
        self._read_web_data()

    def _read_web_data(self):  
        PeMS_file_links = {
                "pems_AllLA_adj_mat.pkl": "https://anl.app.box.com/shared/static/9qc2lc1147xzh8kmq3j4fuo4buiksxua",
                "pems_AllLA_speed.h5": "https://anl.app.box.com/shared/static/crzf75ein8s839de8fklpubauddv1p6w"
        }          
        

        for key in PeMS_file_links.keys():
            
            # Check if file is in data folder from working directory, otherwise download
            if not os.path.isfile(
            os.path.join(self.raw_data_dir,key)
            ):
                print("Downloading to", key, flush=True)
                
                response = requests.get(PeMS_file_links[key], stream=True)
                file_size = int(response.headers.get('content-length', 0))

                with open(os.path.join(self.raw_data_dir, key), "wb") as file, tqdm(
                    total=file_size, unit="B", unit_scale=True, unit_divisor=1024
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=33554432):
                        file.write(chunk)
                        progress_bar.update(len(chunk))
         
    
    def get_index_dataset(self, lags: int = 12, batch_size: int = 64, shuffle: bool = False, allGPU: int = -1, 
                          ratio: Tuple[float, float, float] = (0.7, 0.1, 0.2), world_size: int =-1, ddp_rank: int = -1, 
                          dask_batching: bool = False) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns torch dataloaders using index batching for PeMS dataset.

        Args:
            lags (int, optional): The number of time lags. Defaults to 12.
            batch_size (int, optional): Batch size. Defaults to 64.
            shuffle (bool, optional): If the data should be shuffled. Defaults to False.
            allGPU (int, optional): GPU device ID for performing preprocessing in GPU memory. 
                                    If -1, computation is done on CPU. Defaults to -1.
            world_size (int, optional): The number of workers if DDP is being used. Defaults to -1.
            ddp_rank (int, optional): The DDP rank of the worker if DDP is being used. Defaults to -1.
            ratio (tuple of float, optional): The desired train, validation, and test split ratios, respectively.
                    
        Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        
        A 7-tuple containing:
            - **train_dataLoader** (*torch.utils.data.DataLoader*): Dataloader for the training set.
            - **val_dataLoader** (*torch.utils.data.DataLoader*): Dataloader for the validation set.
            - **test_dataLoader** (*torch.utils.data.DataLoader*): Dataloader for the test set.
            - **edges** (*torch.Tensor*): The graph edges as a 2D matrix, shape `[2, num_edges]`.
            - **edge_weights** (*torch.Tensor*): Each graph edge's weight, shape `[num_edges]`.
            - **means** (*torch.Tensor*): The means of each feature dimension.
            - **stds** (*torch.Tensor*): The standard deviations of each feature dimension.
        """

        # adj matrix setup
        with open(os.path.join(self.raw_data_dir, "pems_AllLA_adj_mat.pkl"), 'rb') as f:
            _, _, adj_mx = pickle.load(f)
        edges, edge_weights = dense_to_sparse(torch.from_numpy(adj_mx))
    
        # setup data
        df = self.pd.read_hdf(os.path.join(self.raw_data_dir, "pems_AllLA_speed.h5"), "df")
        num_samples, num_nodes = df.shape
        

        
        if allGPU != -1:
            data = torch.empty((num_samples, num_nodes, 2), device='cuda',dtype=torch.float)
            data[...,0] = torch.tensor(df.values).to(f"cuda:{allGPU}")
            
        else:
            data = np.expand_dims(df.values, axis=-1)
            data_list = [data[:]]


        
        if allGPU != -1:
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            data[...,1] = torch.squeeze(torch.tensor(np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0)))).to(f"cuda:{allGPU}")
        else:
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        

        # Normalise as in DCRNN paper (via Z-Score Method)
        if allGPU != -1:
            data = data.to(f"cuda:{allGPU}")
            
            means = torch.mean(data, dim=(0, 1), keepdim=True) 
            stds = torch.std(data, dim=(0, 1), keepdim=True) 
            data = (data - means) / stds
            
        else:
            data = np.concatenate(data_list, axis=-1)
            means = np.mean(data, axis=(0, 1))
            stds = np.std(data, axis=(0, 1))
            data = (data - means) / stds 
            
            means = torch.tensor(means,dtype=torch.float)
            stds = torch.tensor(stds,dtype=torch.float)
            
        
        num_samples = data.shape[0]
        x_i = np.arange(num_samples - (2 * lags - 1))
        num_samples = x_i.shape[0]
        num_train = round(num_samples * ratio[0])
        num_test = round(num_samples * ratio[2])
        num_val = num_samples - num_train - num_test

        x_train = x_i[:num_train]
        x_val = x_i[num_train: num_train + num_val]
        x_test = x_i[-num_test:]

        train_dataset = self.IndexDataset(x_train,data,lags,gpu=not (allGPU == -1), lazy=dask_batching)
        val_dataset = self.IndexDataset(x_val,data,lags,gpu=not (allGPU == -1), lazy=dask_batching)
        test_dataset = self.IndexDataset(x_test,data,lags,gpu=not (allGPU == -1),lazy=dask_batching)

        if ddp_rank != -1:
            train_sampler = DistributedSampler(train_dataset,  num_replicas=world_size, rank=ddp_rank, shuffle=shuffle)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
            
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=ddp_rank, shuffle=shuffle)                  
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
            
            test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=ddp_rank, shuffle=shuffle)                  
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)


        return train_dataloader, val_dataloader, test_dataloader, edges, edge_weights, means, stds
