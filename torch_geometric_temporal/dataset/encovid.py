import io
import json
import numpy as np
from six.moves import urllib
from ..signal import DynamicGraphTemporalSignal


class EnglandCovidDatasetLoader(object):
    """A dataset of mobility and history of reported cases of COVID-19
     in England NUTS3 regions, from 3 March to 12 of May. The dataset is 
     segmented in days and the graph is directed and weighted. The graph
     indicates how many people moved from one region to the other each day, 
     based on Facebook Data For Good disease prevention maps. 
     The node features correspond to the number of COVID-19 cases
     in the region in the past **window** days. The task is to predict the
     number of cases in each node after 1 day. For details see this paper: 
     `"Transfer Graph Neural Networks for Pandemic Forecasting." <https://arxiv.org/abs/2009.08388>`_
    """
    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/england_covid.json"
        self._dataset = json.loads(urllib.request.urlopen(url).read())

    def _get_edges(self):
        self._edges = []
        for time in range(self._dataset["time_periods"]-self.lags):      
            self._edges.append(np.array(self._dataset["edge_mapping"]["edge_index"][str(time)]).T)


    def _get_edge_weights(self):
        self._edge_weights = []
        for time in range(self._dataset["time_periods"]-self.lags):      
            self._edge_weights.append(np.array(self._dataset["edge_mapping"]["edge_weight"][str(time)]))

    def _get_targets_and_features(self):

        stacked_target = np.array(self._dataset["y"])
        standardized_target = (stacked_target - np.mean(stacked_target, axis=0)) /(np.std(stacked_target, axis=0)+10**-10)
        self.features = [standardized_target[i:i+self.lags,:].T for i in range(self._dataset["time_periods"]-self.lags)]
        self.targets = [standardized_target[i+self.lags,:].T for i in range(self._dataset["time_periods"]-self.lags)]

    def get_dataset(self, lags: int=8) -> DynamicGraphTemporalSignal:
        """Returning the England COVID19 data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.        
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The England Covid dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = DynamicGraphTemporalSignal(self._edges, self._edge_weights, self.features, self.targets)
        return dataset
