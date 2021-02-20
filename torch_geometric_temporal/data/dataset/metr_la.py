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
