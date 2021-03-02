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
        super(PemsBayDatasetLoader, self).__init()
        self._read_web_data()

    def _read_web_data():
        url = "placeholder"
        pass

    def _get_edges_and_weights(self):
        pass

    def _generate_task(self, num_timesteps_in: int=12, num_timesteps_out: int=2):
        pass

    def get_dataset(self, num_timesteps_in: int=12, num_timesteps_out: int=2):
        pass

if __name__ == '__main__':
    pass



