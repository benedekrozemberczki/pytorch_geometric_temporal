import json
import urllib
import numpy as np
from ..signal import StaticGraphTemporalSignal
from .base import AbstractDataLoader


class WindmillOutputSmallDatasetLoader(AbstractDataLoader):
    """Hourly energy output of windmills from a European country
    for more than 2 years. Vertices represent 11 windmills and
    weighted edges describe the strength of relationships. The target
    variable allows for regression tasks.
    """

    def __init__(self, datadir=None):
        super(WindmillOutputSmallDatasetLoader, self).__init__(
            "windmill_output_small.json", datadir
        )
        self._set_src_prefix("https://graphmining.ai/temporal_datasets/")
        self._dataset = self._load()

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):
        stacked_target = np.stack(self._dataset["block"])
        standardized_target = (stacked_target - np.mean(stacked_target, axis=0)) / (
            np.std(stacked_target, axis=0) + 10**-10
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
