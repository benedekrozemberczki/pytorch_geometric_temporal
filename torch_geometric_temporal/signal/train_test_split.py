from typing import Union, Tuple
from .static_graph_temporal_signal import StaticGraphTemporalSignal
from .dynamic_graph_temporal_signal import DynamicGraphTemporalSignal
from .dynamic_graph_static_signal import DynamicGraphStaticSignal


Discrete_Signal = Union[StaticGraphTemporalSignal, DynamicGraphTemporalSignal, DynamicGraphStaticSignal]

def temporal_signal_split(data_iterator, train_ratio: float=0.8) -> Tuple[Discrete_Signal, Discrete_Signal]:
    r""" Function to split a data iterator according to a fixed ratio.

    Parameters
    ----------
    data_iterator : StaticGraphTemporalSignal, DynamicGraphTemporalSignal or DynamicGraphStaticSignal
        A data iterator to create a temporal train and test split.
    train_ratio : float
        Ratio of training data to be used.

    Returns
    -------
    tuple : 
        Train and test data iterators.

    """
    train_snapshots = int(train_ratio*data_iterator.snapshot_count)

    if type(data_iterator) == StaticGraphTemporalSignal:
        train_iterator = StaticGraphTemporalSignal(data_iterator.edge_index,
                                                   data_iterator.edge_weight,
                                                   data_iterator.features[0:train_snapshots],
                                                   data_iterator.targets[0:train_snapshots])

        test_iterator = StaticGraphTemporalSignal(data_iterator.edge_index,
                                                  data_iterator.edge_weight,
                                                  data_iterator.features[train_snapshots:],
                                                  data_iterator.targets[train_snapshots:])

    elif type(data_iterator) == DynamicGraphTemporalSignal:
        train_iterator = DynamicGraphTemporalSignal(data_iterator.edge_indices[0:train_snapshots],
                                                    data_iterator.edge_weights[0:train_snapshots],
                                                    data_iterator.features[0:train_snapshots],
                                                    data_iterator.targets[0:train_snapshots])

        test_iterator = DynamicGraphTemporalSignal(data_iterator.edge_indices[train_snapshots:],
                                                   data_iterator.edge_weights[train_snapshots:],
                                                   data_iterator.features[train_snapshots:],
                                                   data_iterator.targets[train_snapshots:])
                                                   
    elif type(data_iterator) == DynamicGraphStaticSignal:
        train_iterator = DynamicGraphStaticSignal(data_iterator.edge_indices[0:train_snapshots],
                                                  data_iterator.edge_weights[0:train_snapshots],
                                                  data_iterator.feature,
                                                  data_iterator.targets[0:train_snapshots])

        test_iterator = DynamicGraphStaticSignal(data_iterator.edge_indices[train_snapshots:],
                                                 data_iterator.edge_weights[train_snapshots:],
                                                 data_iterator.feature,
                                                 data_iterator.targets[train_snapshots:])
    return train_iterator, test_iterator
