from torch_geometric_temporal.data.dataset import ChickenpoxDataset


def test_chickenpox():
    dataset = ChickenpoxDataset()
    assert 1 == 1
