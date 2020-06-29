from torch_geometric_temporal.nn import GConvLSTM

def test_gconv_lstm_layer():
    """
    Testing the Dummy Layer.
    """

    layer = GConvLSTM(in_channels=32, out_channels=64, K=5, number_of_nodes=100)

    assert layer.in_channels == 32
    assert layer.out_channels == 64
