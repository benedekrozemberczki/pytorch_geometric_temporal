from typing import Union, Tuple
from torch_geometric_temporal.data.discrete.static_graph_discrete_signal import StaticGraphDiscreteSignal
from torch_geometric_temporal.data.discrete.dynamic_graph_discrete_signal import DynamicGraphDiscreteSignal


Discrete_Signal = Union[StaticGraphDiscreteSignal, DynamicGraphDiscreteSignal]

def discrete_train_test_split(data_iterator, train_ratio: float=0.8) -> Tuple[Discrete_Signal, Discrete_Signal]:
    r""" Function to split a discrete data iterator according to a fixed ratio.

    Parameters
    ----------
    data_iterator : StaticGraphDiscreteSignal or DynamicGraphDiscreteSignal
        A data iterator to create a temporal train and test split.
    train_ratio : float
        Ratio of training data to be used.

    Returns
    -------
    tuple : 
        Train and test data iterators.

    """
    train_snapshots = int(train_ratio*data_iterator.snapshot_count)

    if type(data_iterator) == StaticGraphDiscreteSignal:
        train_iterator = StaticGraphDiscreteSignal(data_iterator.edge_index,
                                                   data_iterator.edge_weight,
                                                   data_iterator.features[0:train_snapshots],
                                                   data_iterator.targets[0:train_snapshots])

        test_iterator = StaticGraphDiscreteSignal(data_iterator.edge_index,
                                                  data_iterator.edge_weight,
                                                  data_iterator.features[train_snapshots:],
                                                  data_iterator.targets[train_snapshots:])

    elif type(data_iterator) == DynamicGraphDiscreteSignal:
        train_iterator = DynamicGraphDiscreteSignal(data_iterator.edge_indices[0:train_snapshots],
                                                    data_iterator.edge_weights[0:train_snapshots],
                                                    data_iterator.features[0:train_snapshots],
                                                    data_iterator.targets[0:train_snapshots])

        test_iterator = DynamicGraphDiscreteSignal(data_iterator.edge_indices[train_snapshots:],
                                                   data_iterator.edge_weights[train_snapshots:],
                                                   data_iterator.features[train_snapshots:],
                                                   data_iterator.targets[train_snapshots:])
    return train_iterator, test_iterator
