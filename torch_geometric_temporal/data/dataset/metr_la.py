import numpy as np
from torch_geometric_temporal.data.discrete.static_graph_discrete_signal import StaticGraphDiscreteSignal


class METRLADatasetLoader(object):
    """A traffic forecasting dataset based on Los Angeles
    Metropolitan traffic conditions. The dataset contains traffic
    readings collected from 207 loop detectors on highways in Los Angeles 
    County in aggregated 5 minute intervals for 4 months between March 2012 
    to June 2012.
    """

    def __init__(self, arg):
        super(MetrlaDatasetLoader, self).__init__()

    def _read_web_data(self):
        url = "placeholder"
        pass

    def _get_edges(self):
        pass

    def _get_edge_weights(self):
        pass

    def _get_features(self):
        pass

    def _get_targets(self):
        pass

    def get_dataset(self) -> StaticGraphDiscreteSignal:
        """Returns data iterator for METR-LA dataset as an instance of the
        static graph discrete signal class.

        Return types:
            * **dataset** *(StaticGraphDiscrete Signal)* - The METR-LA traffic
                forecasting dataset.
        """
        self._get_edges()
        self._get_edge_weights()
        self._get_features()
        self._get_targets()
        dataset = StaticGraphDiscreteSignal(self.edges, self.edge_weights, self.features, self.targets)

        return dataset
