from torch_geometric_temporal.data.dataset import ChickenpoxDatasetLoader


def test_chickenpox():
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()
    for epoch in range(3):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 102)
            assert snapshot.edge_weight.shape == (1, 102)
