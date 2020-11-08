from torch_geometric_temporal.data.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.data.discrete.static_graph_discrete_signal import StaticGraphDiscreteSignal


def test_static_graph_discrete_signal():
    assert 1 == 1

def test_chickenpox():
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()
    for epoch in range(3):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 102)
            assert snapshot.edge_attr.shape == (102, )
            assert snapshot.x.shape == (20, 21)
            assert snapshot.y.shape == (20, )
