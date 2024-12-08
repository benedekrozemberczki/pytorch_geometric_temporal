import json
import urllib
import numpy as np
import os
from ..signal import StaticGraphTemporalSignal
from .base import AbstractDataLoader


class PedalMeDatasetLoader(AbstractDataLoader):
    """A dataset of PedalMe Bicycle deliver orders in London between 2020
    and 2021. We made it public during the development of PyTorch Geometric
    Temporal. The underlying graph is static - vertices are localities and
    edges are spatial_connections. Vertex features are lagged weekly counts of the
    delivery demands (we included 4 lags). The target is the weekly number of
    deliveries the upcoming week. Our dataset consist of more than 30 snapshots (weeks).
    """

    def __init__(self, datadir=None):
        super(PedalMeDatasetLoader, self).__init__("pedalme_london.json", datadir)
        self._dataset = self._load()

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):
        stacked_target = np.array(self._dataset["X"])
        self.features = [
            stacked_target[i : i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]
        self.targets = [
            stacked_target[i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]

    def get_dataset(self, lags: int = 4) -> StaticGraphTemporalSignal:
        """Returning the PedalMe London demand data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The PedalMe dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
