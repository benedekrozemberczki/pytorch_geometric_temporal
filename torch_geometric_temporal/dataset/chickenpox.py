import numpy as np
from ..signal import StaticGraphTemporalSignal
from .base import AbstractDataLoader


class ChickenpoxDatasetLoader(AbstractDataLoader):
    """A dataset of county level chicken pox cases in Hungary between 2004
    and 2014. We made it public during the development of PyTorch Geometric
    Temporal. The underlying graph is static - vertices are counties and
    edges are neighbourhoods. Vertex features are lagged weekly counts of the
    chickenpox cases (we included 4 lags). The target is the weekly number of
    cases for the upcoming week (signed integers). Our dataset consist of more
    than 500 snapshots (weeks).
    """

    def __init__(self, datadir=None):
        super(ChickenpoxDatasetLoader, self).__init__("chickenpox.json", datadir)
        self._dataset = self._load()

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
