import os
import zipfile
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from six.moves import urllib
from torch_geometric_temporal.data.discrete.static_graph_discrete_signal import StaticGraphDiscreteSignal

class PemsBayDatasetLoader(object):
    """A traffic forecasting dataset as described in DCRNN paper
    """

    def __init__(self):
        super(PemsBayDatasetLoader, self).__init__()
        self._read_web_data()

    def _read_web_data(self):
        url = "placeholder"
        
        if (not os.path.isfile("data/PEMS-BAY.zip")):
            urllib.request.urlopen(url).read()

        if (not os.path.isfile("data/pems_adj_mat.npy") or not os.path.isfile("data/pems_node_values.npy")):
            with zipfile.ZipFile("data/PEMS-BAY.zip", "r") as zip_fh:
                zip_fh.extractall("data/")

        A = np.load("data/pems_adj_mat.npy")
        X = np.load("data/pems_node_values.npy").transpose((1,2,0))
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

    def _generate_task(self, num_timesteps_in: int=12, num_timesteps_out: int=2):
        pass
        
    def get_dataset(self, num_timesteps_in: int=12, num_timesteps_out: int=2):
        pass

if __name__ == '__main__':
    from torch_geometric_temporal.data.discrete.static_graph_discrete_signal import StaticGraphDiscreteSignal
    loader = PemsBayDatasetLoader()
    dataset = loader.get_dataset()
