import torch
from torch.utils.data import Dataset
import dask.array as da
import numpy as np



class IndexDataset(Dataset):
    """
    A custom pytorch-compatible dataset that implements index-batching and DDP-index-batching.
    It also supports GPU-index-batching and lazy-index-batching.

    Args:
            indices (array-like): Indices corresponding to the time slicies.
            data (array-like or Dask array): The dataset to be indexed.
            horizon (int): The prediction period for the dataset.
            lazy (bool, optional): Whether to use Dask lazy loading (distribute the data across all workers). Defaults to False.
            gpu (bool, optional): If the data is already on the GPU. Defaults to False.
    """
    def __init__(self, indices, data, horizon, lazy=False, gpu=False):
         self.indices = indices 
         self.data = data
         self.horizon = horizon
         self.lazy = lazy
         self.gpu = gpu
        
    def __len__(self):

        # Return the number of samples
        return self.indices.shape[0]

    def __getitem__(self, x):
        """
        Retrieve a data sample and its corresponding target based on the index.

        Args:
            x (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple (x, y), where `x` is the input sequence and `y` is the target sequence.
        """

        idx = self.indices[x]
        
        # Calculate the offset based on the horizon value
        y_start = idx + self.horizon

        # If the data is already on the gpu (likely due to using index-gpu-preprocessing), return tensor-slice
        if self.gpu:
            return self.data[idx:y_start,...], self.data[y_start:y_start + self.horizon,...]
        
        else:
            # if utilizing DDP-batching, gather the data on to this worker and convert to tensor
            if self.lazy:
                return torch.from_numpy(self.data[idx:y_start,...].compute()),torch.from_numpy(self.data[y_start:y_start + self.horizon,...].compute())
            else:
                return torch.from_numpy(self.data[idx:y_start,...]), torch.from_numpy(self.data[y_start:y_start + self.horizon,...])
    