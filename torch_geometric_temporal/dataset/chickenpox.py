import io
import json
import numpy as np
from six.moves import urllib
from torch_geometric_temporal.data.discrete.static_graph_discrete_signal import StaticGraphDiscreteSignal

class ChickenpoxDatasetLoader(object):
    """A dataset of county level chicken pox cases in Hungary between 2004
    and 2014. We made it public during the development of PyTorch Geometric
    Temporal. The underlying graph is static - vertices are counties and 
    edges are neighbourhoods. Vertex features are lagged weekly counts of the 
    chickenpox cases (we included 4 lags). The target is the weekly number of 
    cases for the upcoming week (signed integers). Our dataset consist of more
    than 500 snapshots (weeks). 
    """
    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/discrete/chickenpox.json"
        self._dataset = json.loads(urllib.request.urlopen(url).read())

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.ones(self._edges.shape[1])

    def _get_features(self):
        self.features = []
        for time in range(self._dataset["time_periods"]):
            self.features.append(np.array(self._dataset[str(time)]["X"]))

    def _get_targets(self):
        self.targets = []
        for time in range(self._dataset["time_periods"]):
            self.targets.append(np.array(self._dataset[str(time)]["y"]))

    def get_dataset(self) -> StaticGraphDiscreteSignal:
        """Returning the Hungarian Chickenpox cases data iterator.

        Return types:
            * **dataset** *(StaticGraphDiscreteSignal)* - The chickenpox dataset.
        """
        self._get_edges()
        self._get_edge_weights()
        self._get_features()
        self._get_targets()
        dataset = StaticGraphDiscreteSignal(self._edges, self._edge_weights, self.features, self.targets)
        return dataset

