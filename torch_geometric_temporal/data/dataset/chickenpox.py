import io
import json
import numpy as np
from six.moves import urllib
from torch_geometric_temporal.data.static_graph_discrete_signal import StaticGraphDiscreteSignal

class ChickenpoxDataset(object):

    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/discrete/chickenpox.json"
        self.dataset = json.loads(urllib.request.urlopen(url).read())

    def _generate_dataset(self):
        edges = np.array(dataset["edges"]).T
        edge_weights = np.ones(edges.shape[1])
        features = []
        targets = []
        for time in range(dataset["time_periods"]):
            features.append(np.array(dataset[str(time)]["y"]))
            targets.append(np.array(dataset[str(time)]["X"]))
 
        
