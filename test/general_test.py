from torch_geometric_temporal.nn import 

def test_gconv_lstm_layer():
    """
    Testing the Dummy Layer.
    """

    layer = GConvLSTM(32, 32)

    assert layer.in_channel == 32
