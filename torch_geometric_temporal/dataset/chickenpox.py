import json
import ssl
import urllib.request
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..signal import StaticGraphTemporalSignal


class ChickenpoxDatasetLoader(object):
    """A dataset of county level chicken pox cases in Hungary between 2004
    and 2014. We made it public during the development of PyTorch Geometric
    Temporal. The underlying graph is static - vertices are counties and
    edges are neighbourhoods. Vertex features are lagged weekly counts of the
    chickenpox cases (we included 4 lags). The target is the weekly number of
    cases for the upcoming week (signed integers). Our dataset consist of more
    than 500 snapshots (weeks).

    Args:
        index (bool, optional): If True, initializes the dataloader to use index-based batching.
            Defaults to False.
    """
    def __init__(self, index=False):
        self._read_web_data()
        self.index = index

        if index == True:
            from ..signal.index_dataset import IndexDataset
            self.IndexDataset = IndexDataset 

    def _read_web_data(self):
        url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/chickenpox.json"

        context = ssl._create_unverified_context()
        self._dataset = json.loads(
            urllib.request.urlopen(url, context=context).read()
        )

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.ones(self._edges.shape[1])

    def _get_targets_and_features(self):
        stacked_target = np.array(self._dataset["FX"])
        self.features = [
            stacked_target[i : i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]
        self.targets = [
            stacked_target[i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]

    def get_dataset(self, lags: int = 4) -> StaticGraphTemporalSignal:
        """Returning the Chickenpox Hungary data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Chickenpox Hungary dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
    
    def get_index_dataset(self, lags=4, batch_size=4, shuffle=False, allGPU=-1, ratio=(0.7, 0.1, 0.2),dask_batching=False):
        """
        Returns torch dataloaders using index batching for Chickenpox Hungary dataset.

        Args:
            lags (int, optional): The number of time lags. Defaults to 4.
            batch_size (int, optional): Batch size. Defaults to 4.
            shuffle (bool, optional): If the data should be shuffled. Defaults to False.
            allGPU (int, optional): GPU device ID for performing preprocessing in GPU memory. 
                                    If -1, computation is done on CPU. Defaults to -1.
            ratio (tuple of float, optional): The desired train, validation, and test split ratios, respectively.

        Returns:
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.Tensor, torch.Tensor]: 
            
            A 5-tuple containing:
                - **train_dataLoader** (*torch.utils.data.DataLoader*): Dataloader for the training set.
                - **val_dataLoader** (*torch.utils.data.DataLoader*): Dataloader for the validation set.
                - **test_dataLoader** (*torch.utils.data.DataLoader*): Dataloader for the test set.
                - **edges** (*torch.Tensor*): The graph edges as a 2D matrix, shape `[2, num_edges]`.
                - **edge_weights** (*torch.Tensor*): Each graph edge's weight, shape `[num_edges]`.
        """
        
        if not self.index:
            raise ValueError("get_index_dataset requires 'index=True' in the constructor.")
        
        data = np.array(self._dataset["FX"])
        edges = torch.tensor(self._dataset["edges"],dtype=torch.int64).T
        edge_weights = torch.ones(edges.shape[1],dtype=torch.float)
        num_samples = data.shape[0]
        
        if allGPU != -1:
            data = torch.tensor(data, dtype=torch.float).to(f"cuda:{allGPU}")
            data = data.unsqueeze(-1)
        else:
            data = np.expand_dims(data, axis=-1)


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
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)


        return train_dataloader, val_dataloader, test_dataloader, edges, edge_weights

