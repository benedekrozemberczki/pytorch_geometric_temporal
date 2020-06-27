from torch_geometric_temporal.nn import DummyLayer

def test_dummy_layer():
    """
    Testing the Dummy Layer.
    """

    layer = DummyLayer()

    assert layer.x == 1
    assert type(layer.x) == int

    layer = DummyLayer(x=2)

    assert layer.x == 2
    assert type(layer.x) == int
