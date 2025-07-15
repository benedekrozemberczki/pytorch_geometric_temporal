import os
import ssl
import urllib.request
import zipfile
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple
from ..signal import StaticGraphTemporalSignal
import requests 
from tqdm import tqdm

class METRLADatasetLoader(object):
    """A traffic forecasting dataset based on Los Angeles
    Metropolitan traffic conditions. The dataset contains traffic
    readings collected from 207 loop detectors on highways in Los Angeles
    County in aggregated 5 minute intervals for 4 months between March 2012
    to June 2012.

    For further details on the version of the sensor network and
    discretization see: `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_
    """

    def __init__(self, raw_data_dir=os.path.join(os.getcwd(), "data"), index: bool = False):
        super(METRLADatasetLoader, self).__init__()
        self.index = index

        self.raw_data_dir = raw_data_dir
        self._read_web_data()

        if index:
            from ..signal.index_dataset import IndexDataset
            self.IndexDataset = IndexDataset 

    def _download_url(self, url, save_path):  # pragma: no cover
        # Check if file is in data folder from working directory, otherwise download
        if not os.path.isfile(
        os.path.join(self.raw_data_dir,save_path)
        ):
            print("Downloading to", save_path, flush=True)
            
            response = requests.get(url, stream=True)
            file_size = int(response.headers.get('content-length', 0))

            with open(os.path.join(self.raw_data_dir, save_path), "wb") as file, tqdm(
                total=file_size, unit="B", unit_scale=True, unit_divisor=1024
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=33554432):
                    file.write(chunk)
                    progress_bar.update(len(chunk))

    def _read_web_data(self):
        url = "https://anl.app.box.com/shared/static/plgsv3te0akmqluiuqva34su60nn93c2"

        # Check if zip file is in data folder from working directory, otherwise download
        if not os.path.isfile(
            os.path.join(self.raw_data_dir, "METR-LA.zip")
        ):  # pragma: no cover
            if not os.path.exists(self.raw_data_dir):
                os.makedirs(self.raw_data_dir)
            self._download_url(url, os.path.join(self.raw_data_dir, "METR-LA.zip"))

        if not os.path.isfile(
            os.path.join(self.raw_data_dir, "adj_mat.npy")
        ) or not os.path.isfile(
            os.path.join(self.raw_data_dir, "node_values.npy")
        ):  # pragma: no cover
            with zipfile.ZipFile(
                os.path.join(self.raw_data_dir, "METR-LA.zip"), "r"
            ) as zip_fh:
                zip_fh.extractall(self.raw_data_dir)
        if not self.index:
            A = np.load(os.path.join(self.raw_data_dir, "adj_mat.npy"))
            X = np.load(os.path.join(self.raw_data_dir, "node_values.npy")).transpose(
                (1, 2, 0)
            )
            X = X.astype(np.float32)

            # Normalise as in DCRNN paper (via Z-Score Method)
            means = np.mean(X, axis=(0, 2))
            X = X - means.reshape(1, -1, 1)
            stds = np.std(X, axis=(0, 2))
            X = X / stds.reshape(1, -1, 1)

            self.A = torch.from_numpy(A)
            self.X = torch.from_numpy(X)

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        """Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting the average traffic speed using num_timesteps_in to predict the
        traffic conditions in the next num_timesteps_out

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, 0, i + num_timesteps_in : j]).numpy())

        self.features = features
        self.targets = target

    def get_dataset(
        self, num_timesteps_in: int = 12, num_timesteps_out: int = 12
    ) -> StaticGraphTemporalSignal:
        """Returns data iterator for METR-LA dataset as an instance of the
        static graph temporal signal class.

        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The METR-LA traffic
                forecasting dataset.
        """
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )

        return dataset
    
    

    def get_index_dataset(self, lags: int = 12, batch_size: int = 64, shuffle: bool = False, allGPU: int = -1, 
                          ratio: Tuple[float, float, float] = (0.7, 0.1, 0.2), world_size: int =-1, ddp_rank: int = -1, 
                          dask_batching: bool = False) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns torch dataloaders using index batching for Metr-LA dataset.

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
        A = np.load(os.path.join(self.raw_data_dir, "adj_mat.npy"))
        edges, edge_weights = dense_to_sparse(torch.from_numpy(A))

        data = np.load(os.path.join(self.raw_data_dir, "node_values.npy")).transpose( (1, 2, 0))
        data = data.astype(np.float32)

        # Normalise as in DCRNN paper (via Z-Score Method)
        if allGPU != -1:
            data = torch.tensor(data,dtype=torch.float).to(f"cuda:{allGPU}")
            means = torch.mean(data, dim=(0, 2), keepdim=True)
            data = data - means

            stds = torch.std(data, dim=(0, 2), keepdim=True)
            data = data / stds
            data = data.permute(2, 0, 1)

            means.squeeze_()
            stds.squeeze_()
        
        else:
            
            means = np.mean(data, axis=(0, 2))
            data = data - means.reshape(1, -1, 1)
            stds = np.std(data, axis=(0, 2))
            data = data / stds.reshape(1, -1, 1)
            data = data.transpose((2, 0, 1))

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

    

