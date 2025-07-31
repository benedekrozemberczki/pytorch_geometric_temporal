from ..signal import StaticGraphTemporalSignal

import io
import ssl
import urllib.request

import numpy as np
import torch


class WaveEquationDatasetLoader(object):
    """Wave equation simulation dataset on German coastline regions.
    
    Contains spatio-temporal dynamics modeling wave propagation phenomena
    such as seismic, acoustic, or water waves. Static graph with 325 nodes
    and 1858 edges across 1728 time steps. For details see this paper:
    `"Synthetic Spatio-Temporal Graphs for Temporal Graph Learning." <https://openreview.net/forum?id=EguDBMechn>`_
    """

    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        signal_url = (
            "https://raw.githubusercontent.com/Jostarndt/"
            "Synthetic_Datasets_for_Temporal_Graphs/main/data/"
            "wave_equation/wave_equation_dataset.npy"
        )
        adj_url = (
            "https://raw.githubusercontent.com/Jostarndt/"
            "Synthetic_Datasets_for_Temporal_Graphs/main/data/"
            "wave_equation/germany_coastline_adjacency.pt"
        )

        context = ssl.create_default_context()

        with urllib.request.urlopen(signal_url, context=context) as fh:
            signal_bytes = fh.read()
        self._dataset = np.load(io.BytesIO(signal_bytes))

        with urllib.request.urlopen(adj_url, context=context) as fh:
            adj_bytes = fh.read()
        dist_tensor = torch.load(io.BytesIO(adj_bytes), map_location="cpu").T
        self._edges = dist_tensor[:2, :].numpy()
        self._edge_weights = dist_tensor[2, :].numpy()

    def _get_targets_and_features(self):
        stacked_data = self._dataset
        
        self.features = [
            stacked_data[i : i + self.lags, :, :].transpose(1, 0, 2).reshape(stacked_data.shape[1], -1)
            for i in range(stacked_data.shape[0] - self.lags)
        ]
        self.targets = [
            stacked_data[i + self.lags, :, :] for i in range(stacked_data.shape[0] - self.lags)
        ]

    def get_dataset(self, lags: int = 4) -> StaticGraphTemporalSignal:
        """Returns the wave equation dataset iterator.

        Args:
            lags (int): Number of time lags. Defaults to 4.
            
        Returns:
            StaticGraphTemporalSignal: The wave equation dataset.
        """
        self.lags = lags
        self._get_targets_and_features()
        
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset 