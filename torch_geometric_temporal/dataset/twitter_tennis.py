import io
import json
import numpy as np
from six.moves import urllib
from ..signal import DynamicGraphTemporalSignal

def transform_degree(x, cutoff=4):
    log_deg = np.ceil(np.log(x+1.0))
    return np.minimum(log_deg, cutoff)

def transform_transitivity(x):
    trans = x * 10
    return np.floor(trans)

def onehot_encoding(x, unique_vals):
    E = np.zeros((len(x), len(unique_vals)))
    for i, val in enumerate(x):
        E[i, unique_vals.index(val)] = 1.0
    return E

def encode_features(X, log_degree_cutoff=4):
    X_arr = np.array(X)
    a = transform_degree(X_arr[:,0], log_degree_cutoff)
    b = transform_transitivity(X_arr[:,1])
    A = onehot_encoding(a, range(log_degree_cutoff+1))
    B = onehot_encoding(b, range(11))
    return np.concatenate((A,B),axis=1)

class TwitterTennisDatasetLoader(object):
    """
    Twitter mention graphs related to major tennis tournaments from 2017. 
    Nodes are Twitter accounts and edges are mentions between them. 
    Each snapshot contains the graph induced by the most popular nodes 
    of the original dataset. Node labels encode the number of mentions 
    received in the original dataset for the next snapshot. Read more 
    on the original Twitter data in the 'Temporal Walk Based Centrality Metric for Graph Streams' paper.
    
    Parameters
    ----------
    event_id : str
        Choose to load the mention network for Roland-Garros 2017 ("rg17") or USOpen 2017 ("uo17")
    N : int <= 1000
        Number of most popular nodes to load. By default N=1000. Each snapshot contains the graph induced by these nodes.
    feature_mode : str
        None : load raw degree and transitivity node features
        "encoded" : load onehot encoded degree and transitivity node features
        "diagonal" : set identity matrix as node features
    target_offset : int
        Set the snapshot offset for the node labels to be predicted. By default node labels for the next snapshot are predicted (target_offset=1).
    """
    def __init__(self, event_id="rg17", N=None, feature_mode="encoded", target_offset=1):
        self.N = N
        self.target_offset = target_offset
        if event_id in ["rg17","uo17"]:
            self.event_id = event_id
        else:
            raise ValueError("Invalid 'event_id'! Choose 'rg17' or 'uo17' to load the Roland-Garros 2017 or the USOpen 2017 Twitter tennis dataset respectively.")
        if feature_mode in [None, "diagonal", "encoded"]:
            self.feature_mode = feature_mode
        else:
            raise ValueError("Choose feature_mode from values [None, 'diagonal', 'encoded'].")
        self._read_web_data()

    def _read_web_data(self):
        fname = "twitter_tennis_%s.json" % self.event_id
        url = "https://raw.githubusercontent.com/ferencberes/pytorch_geometric_temporal/developer/dataset/" + fname 
        self._dataset = json.loads(urllib.request.urlopen(url).read())
        #with open("/home/fberes/git/pytorch_geometric_temporal/dataset/"+fname) as f:
        #    self._dataset = json.load(f)

    def _get_edges(self):
        edge_indices = []
        self.edges = []
        for time in range(self._dataset["time_periods"]):
            E = np.array(self._dataset[str(time)]["edges"])
            if self.N != None:
                selector = np.where((E[:,0] < self.N) & (E[:,1] < self.N))
                E = E[selector]
                edge_indices.append(selector)
            self.edges.append(E.T)
        self.edge_indices = edge_indices

    def _get_edge_weights(self):
        edge_indices = self.edge_indices
        self.edge_weights = []
        for i, time in enumerate(range(self._dataset["time_periods"])):
            W = np.array(self._dataset[str(time)]["weights"])
            if self.N != None:
                W = W[edge_indices[i]]
            self.edge_weights.append(W)

    def _get_features(self):
        self.features = []
        for time in range(self._dataset["time_periods"]):
            X = np.array(self._dataset[str(time)]["X"])
            if self.N != None:
                X = X[:self.N]
            if self.feature_mode == "diagonal":
                X = np.identity(X.shape[0])
            elif self.feature_mode == "encoded":
                X = encode_features(X)                
            self.features.append(X)

    def _get_targets(self):
        self.targets = []
        T = self._dataset["time_periods"]
        for time in range(T):
            # predict node degrees in advance
            snapshot_id = min(time+self.target_offset,T-1)
            y = np.array(self._dataset[str(snapshot_id)]["y"])
            # logarithmic transformation for node degrees
            y = np.log(1.0+y)
            if self.N != None:
                y = y[:self.N]
            self.targets.append(y)

    def get_dataset(self) -> DynamicGraphTemporalSignal:
        """Returning the TennisDataset data iterator.

        Return types:
            * **dataset** *(DynamicGraphTemporalSignal)* - Selected Twitter tennis dataset (Roland-Garros 2017 or USOpen 2017).
        """
        self._get_edges()
        self._get_edge_weights()
        self._get_features()
        self._get_targets()
        dataset = DynamicGraphTemporalSignal(self.edges, self.edge_weights, self.features, self.targets)
        return dataset

