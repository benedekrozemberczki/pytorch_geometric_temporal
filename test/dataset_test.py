import numpy as np
from torch_geometric_temporal.data.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.data.discrete.static_graph_discrete_signal import StaticGraphDiscreteSignal
from torch_geometric_temporal.data.discrete.dynamic_graph_discrete_signal import DynamicGraphDiscreteSignal
from torch_geometric_temporal.data.splitter import discrete_train_test_split
 
def generate_dynamic_graph_discrete_signal(snapshot_count, n_count, feature_count):
    edge_indices = [np.array([edge for nx.gnp_random_graph(n_count, 0.1)) for _ in range(snapshot_count)]
    edge_weights = [np.ones(n_count) for _ in range(snapshot_count)]
    features = [np.random.uniform(0,1,(n_count, feature_count)) for _ in range(snapshot_count)]
    targets = [np.random.uniform(0,1,(n_count,)) for _ in range(snapshot_count)]
    return edge_indices, edge_weights, features, targets



def test_static_graph_discrete_signal():
    dataset = StaticGraphDiscreteSignal(None, None, [None, None],[None, None])
    for snapshot in dataset:
        assert snapshot.edge_index is None
        assert snapshot.edge_attr is None
        assert snapshot.x is None
        assert snapshot.y is None

def test_dynamic_graph_discrete_signal():
    dataset = DynamicGraphDiscreteSignal([None, None], [None, None], [None, None],[None, None])
    for snapshot in dataset:
        assert snapshot.edge_index is None
        assert snapshot.edge_attr is None
        assert snapshot.x is None
        assert snapshot.y is None

def test_static_graph_discrete_signal_typing():
    dataset = StaticGraphDiscreteSignal(None, None, [np.array([1])],[np.array([2])])
    for snapshot in dataset:
        assert snapshot.edge_index is None
        assert snapshot.edge_attr is None
        assert snapshot.x.shape == (1,)
        assert snapshot.y.shape == (1,)

def test_chickenpox():
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()
    for epoch in range(3):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 102)
            assert snapshot.edge_attr.shape == (102, )
            assert snapshot.x.shape == (20, 4)
            assert snapshot.y.shape == (20, )


def test_discrete_train_test_split():
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()
    train_dataset, test_dataset = discrete_train_test_split(dataset, 0.8)

    for epoch in range(2):
        for snapshot in train_dataset:
            assert snapshot.edge_index.shape == (2, 102)
            assert snapshot.edge_attr.shape == (102, )
            assert snapshot.x.shape == (20, 4)
            assert snapshot.y.shape == (20, )

    for epoch in range(2):
        for snapshot in test_dataset:
            assert snapshot.edge_index.shape == (2, 102)
            assert snapshot.edge_attr.shape == (102, )
            assert snapshot.x.shape == (20, 4)
            assert snapshot.y.shape == (20, )


    
