import io
import json
import numpy as np
from six.moves import urllib
from torch_geometric_temporal.data.discrete.static_graph_discrete_signal import StaticGraphDiscreteSignal

class ChickenpoxDatasetLoader(object):

    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/discrete/chickenpox.json"
        self._dataset = json.loads(urllib.request.urlopen(url).read())

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.ones(self._edges.shape[1])

    def get_dataset(self):
        self._get_edges()
        self._get_edge_weights()

        features = []
        targets = []
        for time in range(self._dataset["time_periods"]):
            features.append(np.array(self._dataset[str(time)]["y"]))
            targets.append(np.array(self._dataset[str(time)]["X"]))
        return StaticGraphDiscreteSignal(self._edges, self._edge_weights, features, targets)
 
        
