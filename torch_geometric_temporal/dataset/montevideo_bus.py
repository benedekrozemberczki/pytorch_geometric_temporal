from typing import List
import json
import numpy as np
from six.moves import urllib
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class MontevideoBusDatasetLoader(object):
    """A dataset of inflow passenger at bus stop level from Montevideo city.
    This dataset comprises hourly inflow passenger data at bus stop level for 11 bus lines during
    October 2020 from Montevideo city (Uruguay). The bus lines selected are the ones that carry
    people to the center of the city and they load more than 25% of the total daily inflow traffic.
    Vertices are bus stops, edges are links between bus stops when a bus line connects them and the
    weight represent the road distance. The target is the passenger inflow. This is a curated
    dataset made from different data sources of the Metropolitan Transportation System (STM) of
    Montevideo. These datasets are freely available to anyone in the National Catalog of Open Data
    from the government of Uruguay (https://catalogodatos.gub.uy/).
    """

    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/montevideo_bus.json"
        self._dataset = json.loads(urllib.request.urlopen(url).read())

    def _get_edges(self):
        self._edges = np.array([(d["source"], d["target"]) for d in self._dataset["links"]]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array([(d["weight"]) for d in self._dataset["links"]]).T

    def _get_features(self, feature_vars: List[str] = ["y"]):
        features = []
        for node in self._dataset["nodes"]:
            X = node.get("X")
            for feature_var in feature_vars:
                features.append(np.array(X.get(feature_var)))
        stacked_features = np.stack(features).T
        standardized_features = (stacked_features - np.mean(stacked_features, axis=0)) / np.std(
            stacked_features, axis=0
        )
        self.features = [
            standardized_features[i : i + self.lags, :].T
            for i in range(len(standardized_features) - self.lags)
        ]

    def _get_targets(self, target_var: str = "y"):
        targets = []
        for node in self._dataset["nodes"]:
            y = node.get(target_var)
            targets.append(np.array(y))
        stacked_targets = np.stack(targets).T
        standardized_targets = (stacked_targets - np.mean(stacked_targets, axis=0)) / np.std(
            stacked_targets, axis=0
        )
        self.targets = [
            standardized_targets[i + self.lags, :].T
            for i in range(len(standardized_targets) - self.lags)
        ]

    def get_dataset(
        self, lags: int = 4, target_var: str = "y", feature_vars: List[str] = ["y"]
    ) -> StaticGraphTemporalSignal:
        """Returning the MontevideoBus passenger inflow data iterator.

        Parameters
        ----------
        lags : int, optional
            The number of time lags, by default 4.
        target_var : str, optional
            Target variable name, by default "y".
        feature_vars : List[str], optional
            List of feature variables, by default ["y"].

        Returns
        -------
        StaticGraphTemporalSignal
            The MontevideoBus dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_features(feature_vars)
        self._get_targets(target_var)
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
