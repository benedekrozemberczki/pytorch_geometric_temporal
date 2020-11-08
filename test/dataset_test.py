from torch_geometric_temporal.data.dataset import ChickenpoxDatasetLoader


def test_chickenpox():
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()
    assert dataset.edge_index.shape == (2, 102)
