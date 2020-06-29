from torch_geometric_temporal.nn.conv import GConvLSTM

def test_gconv_lstm_layer():
    """
    Testing the GConvLSTM Layer.
    """

    layer = GConvLSTM(in_channels=32, out_channels=64, K=2, number_of_nodes=100)

    assert layer.in_channels == 32
    assert layer.out_channels == 64
