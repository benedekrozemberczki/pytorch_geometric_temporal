from typing import Union, Tuple

from .static_graph_temporal_signal import StaticGraphTemporalSignal
from .dynamic_graph_temporal_signal import DynamicGraphTemporalSignal
from .dynamic_graph_static_signal import DynamicGraphStaticSignal

from .static_graph_temporal_signal_batch import StaticGraphTemporalSignalBatch
from .dynamic_graph_temporal_signal_batch import DynamicGraphTemporalSignalBatch
from .dynamic_graph_static_signal_batch import DynamicGraphStaticSignalBatch


Discrete_Signal = Union[StaticGraphTemporalSignal, StaticGraphTemporalSignalBatch,
                        DynamicGraphTemporalSignal, DynamicGraphTemporalSignalBatch,
                        DynamicGraphStaticSignal, DynamicGraphStaticSignalBatch]

def temporal_signal_split(data_iterator, train_ratio: float=0.8) -> Tuple[Discrete_Signal, Discrete_Signal]:
    r""" Function to split a data iterator according to a fixed ratio.

    Arg types:
        * **data_iterator** *(Signal Iterator)* - Node features.
        * **train_ratio** *(float)* - Graph edge indices.

    Return types:
        * **(train_iterator, test_iterator)** *(tuple of Signal Iterators)* - Train and test data iterators.
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
                                                 
    if type(data_iterator) == StaticGraphTemporalSignalBatch:
        train_iterator = StaticGraphTemporalSignalBatch(data_iterator.edge_index,
                                                        data_iterator.edge_weight,
                                                        data_iterator.features[0:train_snapshots],
                                                        data_iterator.targets[0:train_snapshots],
                                                        data_iterator.batches)

        test_iterator = StaticGraphTemporalSignalBatch(data_iterator.edge_index,
                                                       data_iterator.edge_weight,
                                                       data_iterator.features[train_snapshots:],
                                                       data_iterator.targets[train_snapshots:],
                                                       data_iterator.batches)

    elif type(data_iterator) == DynamicGraphTemporalSignalBatch:
        train_iterator = DynamicGraphTemporalSignalBatch(data_iterator.edge_indices[0:train_snapshots],
                                                         data_iterator.edge_weights[0:train_snapshots],
                                                         data_iterator.features[0:train_snapshots],
                                                         data_iterator.targets[0:train_snapshots],
                                                         data_iterator.batches[0:train_snapshots])

        test_iterator = DynamicGraphTemporalSignalBatch(data_iterator.edge_indices[train_snapshots:],
                                                        data_iterator.edge_weights[train_snapshots:],
                                                        data_iterator.features[train_snapshots:],
                                                        data_iterator.targets[train_snapshots:],
                                                        data_iterator.batches[train_snapshots:])
                                                   
    elif type(data_iterator) == DynamicGraphStaticSignalBatch:
        train_iterator = DynamicGraphStaticSignalBatch(data_iterator.edge_indices[0:train_snapshots],
                                                       data_iterator.edge_weights[0:train_snapshots],
                                                       data_iterator.feature,
                                                       data_iterator.targets[0:train_snapshots],
                                                       data_iterator.batches[0:train_snapshots:])

        test_iterator = DynamicGraphStaticSignalBatch(data_iterator.edge_indices[train_snapshots:],
                                                      data_iterator.edge_weights[train_snapshots:],
                                                      data_iterator.feature,
                                                      data_iterator.targets[train_snapshots:],
                                                      data_iterator.batches[train_snapshots:]) 
    return train_iterator, test_iterator
