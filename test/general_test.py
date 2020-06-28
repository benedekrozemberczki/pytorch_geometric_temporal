from torch_geometric_temporal.nn import GConvLSTM

def test_gconv_lstm_layer():
    """
    Testing the Dummy Layer.
    """

    layer = GConvLSTM(in_channels=32, out_channels=64)

    assert layer.in_channels == 32
    assert layer.out_channels == 64
