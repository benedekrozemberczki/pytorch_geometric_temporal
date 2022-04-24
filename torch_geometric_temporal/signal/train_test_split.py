from typing import Union, Tuple

from .static_graph_temporal_signal import StaticGraphTemporalSignal
from .dynamic_graph_temporal_signal import DynamicGraphTemporalSignal
from .dynamic_graph_static_signal import DynamicGraphStaticSignal

from .static_graph_temporal_signal_batch import StaticGraphTemporalSignalBatch
from .dynamic_graph_temporal_signal_batch import DynamicGraphTemporalSignalBatch
from .dynamic_graph_static_signal_batch import DynamicGraphStaticSignalBatch

from .static_hetero_graph_temporal_signal import StaticHeteroGraphTemporalSignal
from .dynamic_hetero_graph_temporal_signal import DynamicHeteroGraphTemporalSignal
from .dynamic_hetero_graph_static_signal import DynamicHeteroGraphStaticSignal

from .static_hetero_graph_temporal_signal_batch import StaticHeteroGraphTemporalSignalBatch
from .dynamic_hetero_graph_temporal_signal_batch import DynamicHeteroGraphTemporalSignalBatch
from .dynamic_hetero_graph_static_signal_batch import DynamicHeteroGraphStaticSignalBatch


Discrete_Signal = Union[
    StaticGraphTemporalSignal,
    StaticGraphTemporalSignalBatch,
    DynamicGraphTemporalSignal,
    DynamicGraphTemporalSignalBatch,
    DynamicGraphStaticSignal,
    DynamicGraphStaticSignalBatch,
    StaticHeteroGraphTemporalSignal,
    StaticHeteroGraphTemporalSignalBatch,
    DynamicHeteroGraphTemporalSignal,
    DynamicHeteroGraphTemporalSignalBatch,
    DynamicHeteroGraphStaticSignal,
    DynamicHeteroGraphStaticSignalBatch,
]


def temporal_signal_split(
    data_iterator, train_ratio: float = 0.8
) -> Tuple[Discrete_Signal, Discrete_Signal]:
    r"""Function to split a data iterator according to a fixed ratio.

    Arg types:
        * **data_iterator** *(Signal Iterator)* - Node features.
        * **train_ratio** *(float)* - Graph edge indices.

    Return types:
        * **(train_iterator, test_iterator)** *(tuple of Signal Iterators)* - Train and test data iterators.
    """

    train_snapshots = int(train_ratio * data_iterator.snapshot_count)
    
    train_iterator = data_iterator[0:train_snapshots]
    test_iterator = data_iterator[train_snapshots:]

    return train_iterator, test_iterator
