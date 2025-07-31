from ..signal import StaticGraphTemporalSignal

import io
import ssl
import urllib.request

import numpy as np
import torch


class AdvectionDiffusionDatasetLoader(object):
    """Advection-diffusion equation simulation dataset on German NUTS3 regions.
    
    Contains spatio-temporal dynamics modeling transport and diffusion phenomena
    such as pollutants or heat. Static graph with 400 nodes and 2088 edges
    across 4320 time steps. For details see this paper:
    `"Synthetic Spatio-Temporal Graphs for Temporal Graph Learning." <https://openreview.net/forum?id=EguDBMechn>`_
    """

    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        signal_url = (
            "https://raw.githubusercontent.com/Jostarndt/"
            "Synthetic_Datasets_for_Temporal_Graphs/main/data/"
            "advection_diffusion_equation/advection_diffusion_dataset.npy"
        )
        adj_url = (
            "https://raw.githubusercontent.com/Jostarndt/"
            "Synthetic_Datasets_for_Temporal_Graphs/main/data/"
            "advection_diffusion_equation/nuts3_adjacent_distances.pt"
        )

        context = ssl.create_default_context()

        with urllib.request.urlopen(signal_url, context=context) as fh:
            signal_bytes = fh.read()
        self._dataset = np.load(io.BytesIO(signal_bytes))

        with urllib.request.urlopen(adj_url, context=context) as fh:
            adj_bytes = fh.read()

        dist_tensor = torch.load(io.BytesIO(adj_bytes), map_location="cpu").T

        # First two rows – edge index, third row – edge distances (weights)
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
        """Returns the advection diffusion dataset iterator.

        Args:
            lags (int): Number of time lags. Defaults to 4.
            
        Returns:
            StaticGraphTemporalSignal: The advection diffusion dataset.
        """
        self.lags = lags
        self._get_targets_and_features()
        
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset 