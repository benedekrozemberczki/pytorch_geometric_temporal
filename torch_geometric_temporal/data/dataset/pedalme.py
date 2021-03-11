import io
import json
import numpy as np
from six.moves import urllib
from torch_geometric_temporal.data.discrete.static_graph_discrete_signal import StaticGraphDiscreteSignal

class PedalMeDatasetLoader(object):
    """A dataset of PedalMe Bicycle deliver orders in London between 2020
    and 2021. We made it public during the development of PyTorch Geometric
    Temporal. The underlying graph is static - vertices are localities and 
    edges are spatial_connections. Vertex features are lagged weekly counts of the 
    delivery demands (we included 4 lags). The target is the weekly number of 
    deliveries the upcoming week. Our dataset consist of more than 30 snapshots (weeks). 
    """
    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/discrete/pedalme.json"
        self._dataset = json.loads(urllib.request.urlopen(url).read())

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_features(self):
        self.features = []
        for time in range(self._dataset["time_periods"]):
            self.features.append(np.array(self._dataset[str(time)]["X"]))

    def _get_targets(self):
        self.targets = []
        for time in range(self._dataset["time_periods"]):
            self.targets.append(np.array(self._dataset[str(time)]["y"]))

    def get_dataset(self) -> StaticGraphDiscreteSignal:
        """Returning the PedalMe London demand data iterator.

        Return types:
            * **dataset** *(StaticGraphDiscreteSignal)* - The PedalMe dataset.
        """
        self._get_edges()
        self._get_edge_weights()
        self._get_features()
        self._get_targets()
        dataset = StaticGraphDiscreteSignal(self._edges, self._edge_weights, self.features, self.targets)
        return dataset

