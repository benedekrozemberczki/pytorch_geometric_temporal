from typing import Union 
from torch_geometric_temporal.data.discrete.static_graph_discrete_signal import StaticGraphDiscreteSignal


def discrete_train_test_split(data_iterator, train_ratio: float=0.8):

    train_snapshots = int(train_ratio*len(data_iterator.features))
    test_snapshots = len(data_iterator.features) - train_snapshots

    if type(data_iterator) == StaticGraphDiscreteSignal:
        train_iterator = StaticGraphDiscreteSignal(data_iterator.edge_index, data_iterator.edge_weight, [None],[None])
        test_iterator = StaticGraphDiscreteSignal(data_iterator.edge_index, data_iterator.edge_weight, [None],[None])
    return 1
