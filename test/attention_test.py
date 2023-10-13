import torch
import math
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import barabasi_albert_graph
from torch_geometric.transforms import LaplacianLambdaMax
from torch_geometric_temporal.nn.attention import (
    TemporalConv,
    STConv,
    ASTGCN,
    MSTGCN,
    MTGNN,
    ChebConvAttention,
    AAGCN,
    GraphAAGCN,
    DNNTSP,
)
from torch_geometric_temporal.nn.attention import (
    GMAN,
    SpatioTemporalAttention,
    SpatioTemporalEmbedding,
)


def create_mock_data(number_of_nodes, edge_per_node, in_channels):
    """
    Creating a mock feature matrix and edge index.
    """
    graph = nx.watts_strogatz_graph(number_of_nodes, edge_per_node, 0.5)
    edge_index = torch.LongTensor(np.array([edge for edge in graph.edges()]).T)
    X = torch.FloatTensor(np.random.uniform(-1, 1, (number_of_nodes, in_channels)))
    return X, edge_index


def create_mock_edge_weight(edge_index):
    """
    Creating a mock edge weight tensor.
    """
    return torch.FloatTensor(np.random.uniform(0, 1, (edge_index.shape[1])))


def create_mock_target(number_of_nodes, number_of_classes):
    """
    Creating a mock target vector.
    """
    return torch.LongTensor(
        [np.random.randint(0, number_of_classes - 1) for node in range(number_of_nodes)]
    )


def create_mock_sequence(
    sequence_length, number_of_nodes, edge_per_node, in_channels, number_of_classes
):
    """
    Creating mock sequence data

    Note that this is a static graph discrete signal type sequence
    The target is the "next" item in the sequence
    """
    input_sequence = torch.zeros(sequence_length, number_of_nodes, in_channels)

    X, edge_index = create_mock_data(
        number_of_nodes=number_of_nodes,
        edge_per_node=edge_per_node,
        in_channels=in_channels,
    )
    edge_weight = create_mock_edge_weight(edge_index)
    targets = create_mock_target(number_of_nodes, number_of_classes)

    for t in range(sequence_length):
        input_sequence[t] = X + t

    return input_sequence, targets, edge_index, edge_weight


def create_mock_batch(
    batch_size,
    sequence_length,
    number_of_nodes,
    edge_per_node,
    in_channels,
    number_of_classes,
):
    """
    Creating a mock batch of sequences
    """
    batch = torch.zeros(batch_size, sequence_length, number_of_nodes, in_channels)
    batch_targets = torch.zeros(batch_size, number_of_nodes, dtype=torch.long)

    for b in range(batch_size):
        input_sequence, targets, edge_index, edge_weight = create_mock_sequence(
            sequence_length,
            number_of_nodes,
            edge_per_node,
            in_channels,
            number_of_classes,
        )
        batch[b] = input_sequence
        batch_targets[b] = targets

    return batch, batch_targets, edge_index, edge_weight


def test_temporalconv():
    """
    Testing the temporal block in STGCN
    """
    batch_size = 10
    sequence_length = 5

    number_of_nodes = 300
    in_channels = 100
    edge_per_node = 15
    out_channels = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, _, _, _ = create_mock_batch(
        batch_size,
        sequence_length,
        number_of_nodes,
        edge_per_node,
        in_channels,
        out_channels,
    )

    kernel_size = 3
    temporal_conv = TemporalConv(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
    ).to(device)

    H = temporal_conv(batch.to(device))
    assert H.shape == (
        batch_size,
        sequence_length - (kernel_size - 1),
        number_of_nodes,
        out_channels,
    )


def test_stconv():
    """
    Testing STConv block in STGCN
    """
    batch_size = 10
    sequence_length = 5

    number_of_nodes = 300
    in_channels = 100
    edge_per_node = 15
    out_channels = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, _, edge_index, edge_weight = create_mock_batch(
        batch_size,
        sequence_length,
        number_of_nodes,
        edge_per_node,
        in_channels,
        out_channels,
    )

    kernel_size = 3
    stconv = STConv(
        num_nodes=number_of_nodes,
        in_channels=in_channels,
        hidden_channels=8,
        out_channels=out_channels,
        kernel_size=3,
        K=2,
    ).to(device)
    H = stconv(batch.to(device), edge_index.to(device), edge_weight.to(device))
    assert H.shape == (
        batch_size,
        sequence_length - 2 * (kernel_size - 1),
        number_of_nodes,
        out_channels,
    )


def test_astgcn():
    """
    Testing ASTGCN block and its component ChebConvAttention with changing edge index over time or not
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels, out_channels = (16, 32)
    batch_size = 3
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]]).to(device)
    num_nodes = edge_index.max().item() + 1
    edge_weight = torch.rand(edge_index.size(1)).to(device)
    x = torch.randn((batch_size, num_nodes, in_channels)).to(device)
    attention = torch.nn.functional.softmax(
        torch.rand((batch_size, num_nodes, num_nodes)), dim=1
    ).to(device)

    conv = ChebConvAttention(in_channels, out_channels, K=3, normalization="sym").to(
        device
    )
    assert conv.__repr__() == "ChebConvAttention(16, 32, K=3, normalization=sym)"
    out1 = conv(x, edge_index, attention)
    assert out1.size() == (batch_size, num_nodes, out_channels)
    out2 = conv(x, edge_index, attention, edge_weight)
    assert out2.size() == (batch_size, num_nodes, out_channels)
    out3 = conv(x, edge_index, attention, edge_weight, lambda_max=3.0)
    assert out3.size() == (batch_size, num_nodes, out_channels)

    batch = torch.tensor([0, 0, 1, 1]).to(device)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]).to(device)
    num_nodes = edge_index.max().item() + 1
    edge_weight = torch.rand(edge_index.size(1)).to(device)
    x = torch.randn((batch_size, num_nodes, in_channels)).to(device)
    lambda_max = torch.tensor([2.0, 3.0]).to(device)
    attention = torch.nn.functional.softmax(
        torch.rand((batch_size, num_nodes, num_nodes)), dim=1
    ).to(device)

    out4 = conv(x, edge_index, attention, edge_weight, batch)
    assert out4.size() == (batch_size, num_nodes, out_channels)
    out5 = conv(x, edge_index, attention, edge_weight, batch, lambda_max)
    assert out5.size() == (batch_size, num_nodes, out_channels)

    node_count = 307
    num_classes = 10
    edge_per_node = 15

    num_for_predict = 12
    len_input = 12
    nb_time_strides = 1

    node_features = 2
    nb_block = 2
    K = 3
    nb_chev_filter = 64
    nb_time_filter = 64
    batch_size = 32
    normalization = None
    bias = True

    model = ASTGCN(
        nb_block,
        node_features,
        K,
        nb_chev_filter,
        nb_time_filter,
        nb_time_strides,
        num_for_predict,
        len_input,
        node_count,
        normalization,
        bias,
    ).to(device)
    model2 = ASTGCN(
        nb_block,
        node_features,
        K,
        nb_chev_filter,
        nb_time_filter,
        nb_time_strides,
        num_for_predict,
        len_input,
        node_count,
        "sym",
        False,
    ).to(device)
    model3 = ASTGCN(
        nb_block,
        node_features,
        K,
        nb_chev_filter,
        nb_time_filter,
        nb_time_strides,
        num_for_predict,
        len_input,
        node_count,
        "rw",
        bias,
    ).to(device)
    T = len_input
    x_seq = torch.zeros([batch_size, node_count, node_features, T]).to(device)
    target_seq = torch.zeros([batch_size, node_count, T]).to(device)
    edge_index_seq = []
    for b in range(batch_size):
        for t in range(T):
            x, edge_index = create_mock_data(node_count, edge_per_node, node_features)
            x_seq[b, :, :, t] = x.to(device)
            if b == 0:
                edge_index_seq.append(edge_index.to(device))
            target = create_mock_target(node_count, num_classes).to(device)
            target_seq[b, :, t] = target
    shuffle = True
    train_dataset = torch.utils.data.TensorDataset(x_seq, target_seq)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    for batch_data in train_loader:
        encoder_inputs, _ = batch_data
        outputs0 = model(encoder_inputs, edge_index_seq)
        outputs1 = model(encoder_inputs, edge_index_seq[0])
        outputs2 = model2(encoder_inputs, edge_index_seq[0])
        outputs3 = model2(encoder_inputs, edge_index_seq)
        outputs4 = model3(encoder_inputs, edge_index_seq[0])
        outputs5 = model3(encoder_inputs, edge_index_seq)
    assert outputs0.shape == (batch_size, node_count, num_for_predict)
    assert outputs1.shape == (batch_size, node_count, num_for_predict)
    assert outputs2.shape == (batch_size, node_count, num_for_predict)
    assert outputs3.shape == (batch_size, node_count, num_for_predict)
    assert outputs4.shape == (batch_size, node_count, num_for_predict)
    assert outputs5.shape == (batch_size, node_count, num_for_predict)


def test_mstgcn():
    """
    Testing MSTGCN block with changing edge index over time.
    """
    node_count = 307
    num_classes = 10
    edge_per_node = 15

    num_for_predict = 12
    len_input = 12
    nb_time_strides = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_features = 2
    nb_block = 2
    K = 3
    nb_chev_filter = 64
    nb_time_filter = 64
    batch_size = 32

    model = MSTGCN(
        nb_block,
        node_features,
        K,
        nb_chev_filter,
        nb_time_filter,
        nb_time_strides,
        num_for_predict,
        len_input,
    ).to(device)
    T = len_input
    x_seq = torch.zeros([batch_size, node_count, node_features, T]).to(device)
    target_seq = torch.zeros([batch_size, node_count, T]).to(device)
    edge_index_seq = []

    for b in range(batch_size):
        for t in range(T):
            x, edge_index = create_mock_data(node_count, edge_per_node, node_features)
            x_seq[b, :, :, t] = x.to(device)
            if b == 0:
                edge_index_seq.append(edge_index.to(device))
            target = create_mock_target(node_count, num_classes).to(device)
            target_seq[b, :, t] = target

    shuffle = True
    train_dataset = torch.utils.data.TensorDataset(x_seq, target_seq)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )

    for batch_data in train_loader:
        encoder_inputs, _ = batch_data
        outputs1 = model(encoder_inputs, edge_index_seq)
        outputs2 = model(encoder_inputs, edge_index_seq[0])

    assert outputs1.shape == (batch_size, node_count, num_for_predict)
    assert outputs2.shape == (batch_size, node_count, num_for_predict)


def test_gman():
    """
    Testing GMAN
    """
    L = 1
    K = 8
    d = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_his = 12
    num_pred = 10
    num_nodes = 50
    num_sample = 100
    batch_size = 32
    bn_decay = 0.1
    steps_per_day = 288
    use_bias = True
    mask = False
    trainX = torch.rand(num_sample, num_his, num_nodes).to(device)
    SE, _ = create_mock_data(number_of_nodes=num_nodes, edge_per_node=8, in_channels=64)
    SE = SE.to(device)
    trainTE = (2 * torch.rand((num_sample, num_his + num_pred, 2)) - 1).to(device)
    model = GMAN(
        L,
        K,
        d,
        num_his,
        bn_decay=bn_decay,
        steps_per_day=steps_per_day,
        use_bias=use_bias,
        mask=mask,
    ).to(device)
    L = 2
    model2 = GMAN(
        L,
        K,
        d,
        num_his,
        bn_decay=bn_decay,
        steps_per_day=steps_per_day,
        use_bias=False,
        mask=True,
    ).to(device)

    X = trainX[:batch_size]
    TE = trainTE[:batch_size]
    pred = model(X, SE, TE)
    assert pred.shape == (batch_size, num_pred, num_nodes)
    pred = model2(X, SE, TE)
    assert pred.shape == (batch_size, num_pred, num_nodes)


def test_mtgnn():
    """
    Testing MTGNN block
    """
    gcn_true = True
    build_adj = True
    dropout = 0.3
    subgraph_size = 20
    gcn_depth = 2
    num_nodes = 207
    node_dim = 40
    dilation_exponential = 1
    conv_channels = 32
    residual_channels = 32
    skip_channels = 64
    end_channels = 128
    in_dim = 2
    seq_in_len = 12
    seq_out_len = 10
    layers = 3
    batch_size = 16
    propalpha = 0.05
    tanhalpha = 3
    num_split = 1
    num_edges = 10
    kernel_size = 7
    kernel_set = [2, 3, 6, 7]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = barabasi_albert_graph(num_nodes, num_edges).to(device)
    A_tilde = (
        torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.size(1)).to(device),
            (num_nodes, num_nodes),
        ).to_dense()
    ).to(device)
    model = MTGNN(
        gcn_true=gcn_true,
        build_adj=build_adj,
        gcn_depth=gcn_depth,
        num_nodes=num_nodes,
        kernel_size=kernel_size,
        kernel_set=kernel_set,
        dropout=dropout,
        subgraph_size=subgraph_size,
        node_dim=node_dim,
        dilation_exponential=dilation_exponential,
        conv_channels=conv_channels,
        residual_channels=residual_channels,
        skip_channels=skip_channels,
        end_channels=end_channels,
        seq_length=seq_in_len,
        in_dim=in_dim,
        out_dim=seq_out_len,
        layers=layers,
        propalpha=propalpha,
        tanhalpha=tanhalpha,
        layer_norm_affline=True,
    ).to(device)
    xd = 8
    FE = torch.rand(num_nodes, xd).to(device)
    model2 = MTGNN(
        gcn_true=gcn_true,
        build_adj=build_adj,
        gcn_depth=gcn_depth,
        num_nodes=num_nodes,
        kernel_size=kernel_size,
        kernel_set=kernel_set,
        dropout=dropout,
        subgraph_size=subgraph_size,
        node_dim=node_dim,
        dilation_exponential=dilation_exponential,
        conv_channels=conv_channels,
        residual_channels=residual_channels,
        skip_channels=skip_channels,
        end_channels=end_channels,
        seq_length=seq_in_len,
        in_dim=in_dim,
        out_dim=seq_out_len,
        layers=layers,
        propalpha=propalpha,
        tanhalpha=tanhalpha,
        layer_norm_affline=True,
        xd=xd,
    ).to(device)

    model3 = MTGNN(
        gcn_true=gcn_true,
        build_adj=False,
        gcn_depth=gcn_depth,
        num_nodes=num_nodes,
        kernel_size=kernel_size,
        kernel_set=kernel_set,
        dropout=dropout,
        subgraph_size=subgraph_size,
        node_dim=node_dim,
        dilation_exponential=dilation_exponential,
        conv_channels=conv_channels,
        residual_channels=residual_channels,
        skip_channels=skip_channels,
        end_channels=end_channels,
        seq_length=seq_in_len,
        in_dim=in_dim,
        out_dim=seq_out_len,
        layers=layers,
        propalpha=propalpha,
        tanhalpha=tanhalpha,
        layer_norm_affline=True,
        xd=xd,
    ).to(device)
    model4 = MTGNN(
        gcn_true=False,
        build_adj=build_adj,
        gcn_depth=gcn_depth,
        num_nodes=num_nodes,
        kernel_size=kernel_size,
        kernel_set=kernel_set,
        dropout=dropout,
        subgraph_size=subgraph_size,
        node_dim=node_dim,
        dilation_exponential=2,
        conv_channels=conv_channels,
        residual_channels=residual_channels,
        skip_channels=skip_channels,
        end_channels=end_channels,
        seq_length=seq_in_len,
        in_dim=in_dim,
        out_dim=seq_out_len,
        layers=layers,
        propalpha=propalpha,
        tanhalpha=tanhalpha,
        layer_norm_affline=False,
    ).to(device)
    trainx = (2 * torch.rand(batch_size, seq_in_len, num_nodes, in_dim) - 1).to(device)
    trainx = trainx.transpose(1, 3)
    perm = torch.randperm(num_nodes).to(device)
    num_sub = int(num_nodes / num_split)
    for j in range(num_split):
        if j != num_split - 1:
            id = perm[j * num_sub: (j + 1) * num_sub]
        else:
            id = perm[j * num_sub:]
        tx = trainx[:, :, id, :]
        output = model(tx, A_tilde, idx=id)
        output = output.transpose(1, 3)
        assert output.shape == (batch_size, 1, num_nodes, seq_out_len)
        output2 = model2(tx, A_tilde, FE=FE)
        output2 = output2.transpose(1, 3)
        assert output2.shape == (batch_size, 1, num_nodes, seq_out_len)
        output3 = model3(tx, A_tilde, FE=FE)
        output3 = output3.transpose(1, 3)
        assert output3.shape == (batch_size, 1, num_nodes, seq_out_len)
        output4 = model4(tx, A_tilde)
        output4 = output4.transpose(1, 3)
        assert output4.shape == (batch_size, 1, num_nodes, seq_out_len)

    seq_in_len = 24
    seq_out_len = 5
    model = MTGNN(
        gcn_true=gcn_true,
        build_adj=build_adj,
        gcn_depth=gcn_depth,
        num_nodes=num_nodes,
        kernel_size=kernel_size,
        kernel_set=kernel_set,
        dropout=dropout,
        subgraph_size=subgraph_size,
        node_dim=node_dim,
        dilation_exponential=dilation_exponential,
        conv_channels=conv_channels,
        residual_channels=residual_channels,
        skip_channels=skip_channels,
        end_channels=end_channels,
        seq_length=seq_in_len,
        in_dim=in_dim,
        out_dim=seq_out_len,
        layers=layers,
        propalpha=propalpha,
        tanhalpha=tanhalpha,
        layer_norm_affline=False,
    ).to(device)
    dilation_exponential = 2
    build_adj = False
    model2 = MTGNN(
        gcn_true=gcn_true,
        build_adj=build_adj,
        gcn_depth=gcn_depth,
        num_nodes=num_nodes,
        kernel_size=kernel_size,
        kernel_set=kernel_set,
        dropout=dropout,
        subgraph_size=subgraph_size,
        node_dim=node_dim,
        dilation_exponential=dilation_exponential,
        conv_channels=conv_channels,
        residual_channels=residual_channels,
        skip_channels=skip_channels,
        end_channels=end_channels,
        seq_length=seq_in_len,
        in_dim=in_dim,
        out_dim=seq_out_len,
        layers=layers,
        propalpha=propalpha,
        tanhalpha=tanhalpha,
        layer_norm_affline=True,
        xd=xd,
    ).to(device)
    model3 = MTGNN(
        gcn_true=False,
        build_adj=build_adj,
        gcn_depth=gcn_depth,
        num_nodes=num_nodes,
        kernel_size=kernel_size,
        kernel_set=kernel_set,
        dropout=dropout,
        subgraph_size=subgraph_size,
        node_dim=node_dim,
        dilation_exponential=dilation_exponential,
        conv_channels=conv_channels,
        residual_channels=residual_channels,
        skip_channels=skip_channels,
        end_channels=end_channels,
        seq_length=seq_in_len,
        in_dim=in_dim,
        out_dim=seq_out_len,
        layers=layers,
        propalpha=propalpha,
        tanhalpha=tanhalpha,
        layer_norm_affline=False,
    ).to(device)
    trainx = (2 * torch.rand(batch_size, seq_in_len, num_nodes, in_dim) - 1).to(device)
    trainx = trainx.transpose(1, 3)
    for j in range(num_split):
        if j != num_split - 1:
            id = perm[j * num_sub: (j + 1) * num_sub]
        else:
            id = perm[j * num_sub:]
        tx = trainx[:, :, id, :]
        output = model(tx, A_tilde, idx=id)
        output = output.transpose(1, 3)
        assert output.shape == (batch_size, 1, num_nodes, seq_out_len)
        output2 = model2(tx, A_tilde, idx=id, FE=FE)
        output2 = output2.transpose(1, 3)
        assert output2.shape == (batch_size, 1, num_nodes, seq_out_len)
        output3 = model3(tx, A_tilde)
        output3 = output3.transpose(1, 3)
        assert output3.shape == (batch_size, 1, num_nodes, seq_out_len)


def test_tsagcn():
    """
    Testing 2s-AGCN unit
    """
    batch_size = 10
    sequence_length = 5

    number_of_nodes = 300
    in_channels = 100
    edge_per_node = 15
    out_channels = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, _, edge_index, edge_weight = create_mock_batch(
        batch_size,
        sequence_length,
        number_of_nodes,
        edge_per_node,
        in_channels,
        out_channels,
    )
    # (bs, seq, nodes, f_in) -> (bs, f_in, seq, nodes)
    # also be sure to pass in a contiguous tensor (the created in create_mock_batch() is not!)
    batch = batch.permute(0, 3, 1, 2).contiguous()
    edge_index = edge_index.to(device)

    stride = 2
    aagcn_adaptive = AAGCN(
        num_nodes=number_of_nodes,
        edge_index=edge_index,
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        adaptive=True,
    ).to(device)
    aagcn_non_adaptive = AAGCN(
        num_nodes=number_of_nodes,
        edge_index=edge_index,
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        adaptive=False,
    ).to(device)
    A = aagcn_adaptive.A

    x_mock = batch.to(device)
    H_adaptive = aagcn_adaptive(x_mock)
    H_non_adaptive = aagcn_non_adaptive(x_mock)

    assert H_adaptive.shape == (
        batch_size,
        out_channels,
        math.ceil(sequence_length / stride),
        number_of_nodes,
    )
    assert H_non_adaptive.shape == (
        batch_size,
        out_channels,
        math.ceil(sequence_length / stride),
        number_of_nodes,
    )
    assert A.shape == (3, number_of_nodes, number_of_nodes)


def test_dnntsp():

    model = DNNTSP(items_total=100, item_embedding_dim=16, n_heads=4)

    g = nx.watts_strogatz_graph(1000, 10, 0.4)

    edges = torch.LongTensor(np.array([[edge[0], edge[1]] for edge in g.edges()])).T

    edge_weight = torch.FloatTensor(np.random.uniform(0, 1, (5000,)))

    node_features = torch.FloatTensor(np.random.uniform(0, 1, (1000, 16)))

    z = model(node_features, edges, edge_weight)

    assert z.shape == (10, 100, 16)
