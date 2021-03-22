import io
import json
import numpy as np
from six.moves import urllib
from torch_geometric_temporal.data.discrete.static_graph_discrete_signal import StaticGraphDiscreteSignal

class WikiMathsDatasetLoader(object):
    """A dataset of vital mathematics articles from Wikipedia. We made it 
    public during the development of PyTorch Geometric Temporal. The 
    underlying graph is static - vertices are Wikipedia pages and edges are 
    links between them. The graph is directed and weighted. Weights represent
    the number of links found at the source Wikipedia page linking to the target
    Wikipedia page. The target is the daily user visits to the Wikipedia pages
    between march 16. 2019 and march 15. 2021. which results in 731 periods.
    """
    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/discrete/wikivital_mathematics.json"
        self._dataset = json.loads(urllib.request.urlopen(url).read())

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):
        self.targets = []
        for time in range(self._dataset["time_periods"]):
            self.targets.append(np.array(self._dataset[str(time)]["y"]))
        stacked_target = np.stack(self.targets)
        standardized_target = (stacked_target - np.mean(stacked_target, axis=0)) / np.std(stacked_target, axis=0)
        self.features = [standardized_target[i:i+self.lags,:].T for i in range(len(self.targets)-self.lags)]
        self.targets = self.targets[self.lags:]

    def get_dataset(self, lags: int=8) -> StaticGraphDiscreteSignal:
        """Returning the Wikipedia Vital Mathematics data iterator.

        Return types:
            * **dataset** *(StaticGraphDiscreteSignal)* - The Wiki Maths dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphDiscreteSignal(self._edges, self._edge_weights, self.features, self.targets)
        return dataset

