import json
import ssl
import urllib.request
import numpy as np
from ..signal import StaticGraphTemporalSignal
import torch
from torch.utils.data import DataLoader
import os
import requests 
from tqdm import tqdm

class WindmillOutputLargeDatasetLoader(object):
    """Hourly energy output of windmills from a European country
    for more than 2 years. Vertices represent 319 windmills and
    weighted edges describe the strength of relationships. The target
    variable allows for regression tasks.

    Args:
        index (bool, optional): If True, initializes the dataloader to use index-based batching.
            Defaults to False.
    """

    def __init__(self, raw_data_dir=os.path.join(os.getcwd(), "data"), index=False):
        self.raw_data_dir = raw_data_dir
        self._read_web_data()
        self.index = index
        if index:
            from ..signal.index_dataset import IndexDataset
            self.IndexDataset = IndexDataset 


    def _read_web_data(self):
        os.makedirs(self.raw_data_dir, exist_ok=True)
        if not os.path.isfile(
            os.path.join(self.raw_data_dir, "windmill_output.json")
        ):  
            url = "https://anl.app.box.com/shared/static/wgwb75lt3ty3pv5a15y9bilx1mjhcq59"
            save_path = f"{self.raw_data_dir}/windmill_output.json"
            print("Downloading to", save_path, flush=True)
            
            response = requests.get(url, stream=True)
            file_size = int(response.headers.get('content-length', 0))

            with open(os.path.join(self.raw_data_dir, save_path), "wb") as file, tqdm(
                total=file_size, unit="B", unit_scale=True, unit_divisor=1024
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=33554432):
                    file.write(chunk)
                    progress_bar.update(len(chunk))
       
        with open(f"{self.raw_data_dir}/windmill_output.json", 'r') as f:
            self._dataset = json.load(f)

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):
        stacked_target = np.stack(self._dataset["block"])
        standardized_target = (stacked_target - np.mean(stacked_target, axis=0)) / (
            np.std(stacked_target, axis=0) + 10 ** -10
        )
        self.features = [
            standardized_target[i : i + self.lags, :].T
            for i in range(standardized_target.shape[0] - self.lags)
        ]
        self.targets = [
            standardized_target[i + self.lags, :].T
            for i in range(standardized_target.shape[0] - self.lags)
        ]

    def get_dataset(self, lags: int = 8) -> StaticGraphTemporalSignal:
        """Returning the Windmill Output data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Windmill Output dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset

    def get_index_dataset(self, lags=8, batch_size=64, shuffle=False, allGPU=-1, ratio=(0.7, 0.1, 0.2),dask_batching=False):
        """
        Returns torch dataloaders using index batching for WindmillLarge dataset.

        Args:
            lags (int, optional): The number of time lags. Defaults to 8.
            batch_size (int, optional): Batch size. Defaults to 64.
            shuffle (bool, optional): If the data should be shuffled. Defaults to False.
            allGPU (int, optional): GPU device ID for performing preprocessing in GPU memory. 
                                    If -1, computation is done on CPU. Defaults to -1.
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
        if not self.index:
            raise ValueError("get_index_dataset requires 'index=True' in the constructor.")
        # adj matrix setup
        edges = torch.tensor(self._dataset["edges"], dtype=torch.int64).T
        edge_weights = torch.tensor(self._dataset["weights"], dtype=torch.float)
        
        # data preprocessing
        data = np.stack(self._dataset["block"])
        num_samples = data.shape[0]    
        
        if allGPU != -1:
            data = torch.tensor(data, dtype=torch.float).to(f"cuda:{allGPU}")
            mean = torch.mean(data, axis=0)
            std = torch.std(data, axis=0) + 10 ** -10
            data  = (data - mean) / std
            data = data.unsqueeze(-1)
        else:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0) + 10 ** -10
            data  = (data - mean) / std
            data = np.expand_dims(data, axis=-1)
            
            mean = torch.tensor(mean,dtype=torch.float)
            std = torch.tensor(std,dtype=torch.float) 

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

        return train_dataloader, val_dataloader, test_dataloader, edges, edge_weights, mean, std