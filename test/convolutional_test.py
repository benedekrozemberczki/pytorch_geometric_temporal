import torch
import numpy as np
import networkx as nx
from torch_geometric_temporal.nn.convolutional import TemporalConv, STConv, ASTGCN, MSTGCN, ChebConvAtt, MTGNN, graph_constructor, mixprop
from torch_geometric.transforms import LaplacianLambdaMax
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix

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
    return torch.LongTensor([np.random.randint(0, number_of_classes-1) for node in range(number_of_nodes)])

def create_mock_sequence(sequence_length, number_of_nodes, edge_per_node, in_channels, number_of_classes):
    """
    Creating mock sequence data
    
    Note that this is a static graph discrete signal type sequence
    The target is the "next" item in the sequence
    """

    input_sequence = torch.zeros(sequence_length, number_of_nodes, in_channels)
    
    X, edge_index = create_mock_data(number_of_nodes=number_of_nodes, edge_per_node=edge_per_node, in_channels=in_channels)
    edge_weight = create_mock_edge_weight(edge_index)
    targets = create_mock_target(number_of_nodes, number_of_classes)

    for t in range(sequence_length):
        input_sequence[t] = X+t

    return input_sequence, targets, edge_index, edge_weight

def create_mock_batch(batch_size, sequence_length, number_of_nodes, edge_per_node, in_channels, number_of_classes):
    """
    Creating a mock batch of sequences
    """
    batch = torch.zeros(batch_size, sequence_length, number_of_nodes, in_channels)
    batch_targets = torch.zeros(batch_size, number_of_nodes, dtype=torch.long)
    
    for b in range(batch_size):
        input_sequence, targets, edge_index, edge_weight = create_mock_sequence(sequence_length, number_of_nodes, edge_per_node, in_channels, number_of_classes)
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
    batch, batch_targets, edge_index, edge_weight = create_mock_batch(batch_size, sequence_length, number_of_nodes, edge_per_node, in_channels, out_channels)

    kernel_size = 3
    temporal_conv = TemporalConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    H = temporal_conv(batch)
    assert H.shape == (batch_size, sequence_length-(kernel_size-1), number_of_nodes, out_channels)    

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
    batch, batch_targets, edge_index, edge_weight = create_mock_batch(batch_size, sequence_length, number_of_nodes, edge_per_node, in_channels, out_channels)

    kernel_size = 3
    stconv = STConv(num_nodes=number_of_nodes, in_channels=in_channels, hidden_channels=8, out_channels=out_channels, kernel_size=3, K=2)
    H = stconv(batch, edge_index, edge_weight)
    assert H.shape == (batch_size, sequence_length-2*(kernel_size-1), number_of_nodes, out_channels)
def test_mtgnn_layers():
    """
    Testing MTGNN layers
    """
    dropout = 0.3
    subgraph_size = 20
    gcn_depth = 2
    num_nodes = 207
    node_dim = 40
    conv_channels = 32
    residual_channels = 32
    skip_channels = 64
    in_dim = 2
    seq_in_len = 12
    batch_size = 16
    propalpha = 0.05
    tanhalpha = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_size = batch_size
    x_all = torch.zeros(total_size,seq_in_len,num_nodes,in_dim)
    for i in range(total_size):
        for j in range(seq_in_len):
            x, _ = create_mock_data(number_of_nodes=num_nodes, edge_per_node=8, in_channels=in_dim)
            x_all[i,j] = x
    # define model and optimizer
    start_conv = torch.nn.Conv2d(in_channels=in_dim,
                        out_channels=residual_channels,
                        kernel_size=(1, 1)).to(device)
    gc = graph_constructor(num_nodes, subgraph_size, node_dim, alpha=tanhalpha, static_feat=None).to(device)
    adp = gc(torch.arange(num_nodes))
    assert adp.shape == (num_nodes, num_nodes)
    x_tmp = start_conv(x_all[:batch_size].transpose(1,3))
    model = mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha)
    mixprop_output = model(x_tmp,adp)
    assert mixprop_output.shape == (batch_size, residual_channels, num_nodes, seq_in_len)

def test_mtgnn():
    """
    Testing MTGNN block
    """
    gcn_true = True
    buildA_true = True
    cl = True
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
    seq_out_len = 12
    layers = 3
    batch_size = 16
    learning_rate = 0.001
    weight_decay = 0.00001
    clip = 5
    step_size1 = 2500
    step_size2 = 100
    epochs = 3
    seed = 101
    propalpha = 0.05
    tanhalpha = 3
    num_split = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(3)
    x, edge_index = create_mock_data(number_of_nodes=num_nodes, edge_per_node=8, in_channels=in_dim)
    mock_adj = to_scipy_sparse_matrix(edge_index)
    predefined_A = torch.tensor(mock_adj.toarray()).to(device)
    x_all = torch.zeros(batch_size,seq_in_len,num_nodes,in_dim)
    for i in range(batch_size):
        for j in range(seq_in_len):
            x, _ = create_mock_data(number_of_nodes=num_nodes, edge_per_node=8, in_channels=in_dim)
            x_all[i,j] = x
    # define model and optimizer
    model = MTGNN(gcn_true, buildA_true, gcn_depth, num_nodes,
                predefined_A=predefined_A,
                dropout=dropout, subgraph_size=subgraph_size,
                node_dim=node_dim,
                dilation_exponential=dilation_exponential,
                conv_channels=conv_channels, residual_channels=residual_channels,
                skip_channels=skip_channels, end_channels= end_channels,
                seq_length=seq_in_len, in_dim=in_dim, out_dim=seq_out_len,
                layers=layers, propalpha=propalpha, tanhalpha=tanhalpha, layer_norm_affline=True)
    trainx = torch.Tensor(x_all).to(device)
    trainx= trainx.transpose(1, 3)
    perm = np.random.permutation(range(num_nodes))
    num_sub = int(num_nodes/num_split) # number of nodes in each sudgraph
    for j in range(num_split):
        if j != num_split-1:
            id = perm[j * num_sub:(j + 1) * num_sub]
        else:
            id = perm[j * num_sub:]
        id = torch.tensor(id).to(device) # a permutation of node id
        tx = trainx[:, :, id, :]
        output = model(tx, idx=id)
        output = output.transpose(1,3)
        assert output.shape == (batch_size, 1, num_nodes, seq_out_len)

def test_astgcn():
    """
    Testing ASTGCN block
    """
    node_count = 307
    num_classes = 10
    edge_per_node = 15


    num_of_vertices = node_count # 307
    num_for_predict = 12
    len_input = 12
    nb_time_strides = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_features = 1
    nb_block = 2
    K = 3
    nb_chev_filter = 64
    nb_time_filter = 64
    batch_size = 32

    x, edge_index = create_mock_data(node_count, edge_per_node, node_features)
    model = ASTGCN(nb_block, node_features, K, nb_chev_filter, nb_time_filter, nb_time_strides, num_for_predict, len_input, node_count).to(device)
    T = len_input
    x_seq = torch.zeros([batch_size,node_count, node_features,T]).to(device)
    target_seq = torch.zeros([batch_size,node_count,T]).to(device)
    for b in range(batch_size):
        for t in range(T):
            x, edge_index = create_mock_data(node_count, edge_per_node, node_features)
            x_seq[b,:,:,t] = x
            target = create_mock_target(node_count, num_classes)
            target_seq[b,:,t] = target
    shuffle = True
    train_dataset = torch.utils.data.TensorDataset(x_seq, target_seq)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    criterion = torch.nn.MSELoss().to(device)
    for batch_data in train_loader:
        encoder_inputs, labels = batch_data
        outputs = model(encoder_inputs, edge_index)
    assert outputs.shape == (batch_size, node_count, num_for_predict)

def test_mstgcn():
    """
    Testing MSTGCN block
    """
    node_count = 307
    num_classes = 10
    edge_per_node = 15


    num_of_vertices = node_count # 307
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

    x, edge_index = create_mock_data(node_count, edge_per_node, node_features)
    model = MSTGCN(nb_block, node_features, K, nb_chev_filter, nb_time_filter, nb_time_strides,num_for_predict, len_input)
    T = len_input
    x_seq = torch.zeros([batch_size,node_count, node_features,T]).to(device)
    target_seq = torch.zeros([batch_size,node_count,T]).to(device)
    for b in range(batch_size):
        for t in range(T):
            x, edge_index = create_mock_data(node_count, edge_per_node, node_features)
            x_seq[b,:,:,t] = x
            target = create_mock_target(node_count, num_classes)
            target_seq[b,:,t] = target
    shuffle = True
    train_dataset = torch.utils.data.TensorDataset(x_seq, target_seq)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    criterion = torch.nn.MSELoss().to(device)
    for batch_data in train_loader:
        encoder_inputs, labels = batch_data
        outputs = model(encoder_inputs, edge_index)
    assert outputs.shape == (batch_size, node_count, num_for_predict)

def test_astgcn_change_edge_index():
    """
    Testing ASTGCN block with changing edge index over time
    """
    node_count = 307
    num_classes = 10
    edge_per_node = 15


    num_of_vertices = node_count # 307
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

    x, edge_index = create_mock_data(node_count, edge_per_node, node_features)
    model = ASTGCN(nb_block, node_features, K, nb_chev_filter, nb_time_filter, nb_time_strides, num_for_predict, len_input, node_count).to(device)
    T = len_input
    x_seq = torch.zeros([batch_size,node_count, node_features,T]).to(device)
    target_seq = torch.zeros([batch_size,node_count,T]).to(device)
    edge_index_seq = []
    for b in range(batch_size):
        for t in range(T):
            x, edge_index = create_mock_data(node_count, edge_per_node, node_features)
            x_seq[b,:,:,t] = x
            if b == 0:
                edge_index_seq.append(edge_index)
            target = create_mock_target(node_count, num_classes)
            target_seq[b,:,t] = target
    shuffle = True
    train_dataset = torch.utils.data.TensorDataset(x_seq, target_seq)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    criterion = torch.nn.MSELoss().to(device)
    for batch_data in train_loader:
        encoder_inputs, labels = batch_data
        outputs = model(encoder_inputs, edge_index_seq)
    assert outputs.shape == (batch_size, node_count, num_for_predict)

def test_mstgcn_change_edge_index():
    """
    Testing MSTGCN block with changing edge index over time
    """
    node_count = 307
    num_classes = 10
    edge_per_node = 15


    num_of_vertices = node_count # 307
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

    x, edge_index = create_mock_data(node_count, edge_per_node, node_features)
    model = MSTGCN(nb_block, node_features, K, nb_chev_filter, nb_time_filter, nb_time_strides,num_for_predict, len_input)
    T = len_input
    x_seq = torch.zeros([batch_size,node_count, node_features,T]).to(device)
    target_seq = torch.zeros([batch_size,node_count,T]).to(device)
    edge_index_seq = []
    for b in range(batch_size):
        for t in range(T):
            x, edge_index = create_mock_data(node_count, edge_per_node, node_features)
            x_seq[b,:,:,t] = x
            if b == 0:
                edge_index_seq.append(edge_index)
            target = create_mock_target(node_count, num_classes)
            target_seq[b,:,t] = target
    shuffle = True
    train_dataset = torch.utils.data.TensorDataset(x_seq, target_seq)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    criterion = torch.nn.MSELoss().to(device)
    for batch_data in train_loader:
        encoder_inputs, labels = batch_data
        outputs = model(encoder_inputs, edge_index_seq)
    assert outputs.shape == (batch_size, node_count, num_for_predict)

def test_chebconvatt():
    """
    Testing ChebCOnvAtt block
    """
    node_count = 307
    num_classes = 10
    edge_per_node = 15


    len_input = 12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_features = 2
    K = 3
    nb_chev_filter = 64
    batch_size = 32

    x, edge_index = create_mock_data(node_count, edge_per_node, node_features)
    model = ChebConvAtt(node_features, nb_chev_filter, K)
    spatial_attention = torch.rand(batch_size,node_count,node_count)
    spatial_attention = torch.nn.functional.softmax(spatial_attention, dim=1)
    model.train()
    T = len_input
    x_seq = torch.zeros([batch_size,node_count, node_features,T]).to(device)
    target_seq = torch.zeros([batch_size,node_count,T]).to(device)
    for b in range(batch_size):
        for t in range(T):
            x, edge_index = create_mock_data(node_count, edge_per_node, node_features)
            x_seq[b,:,:,t] = x
            target = create_mock_target(node_count, num_classes)
            target_seq[b,:,t] = target
    shuffle = True
    train_dataset = torch.utils.data.TensorDataset(x_seq, target_seq)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    for batch_data in train_loader:
        encoder_inputs, labels = batch_data
        data = Data(edge_index=edge_index, edge_attr=None, num_nodes=node_count)
        lambda_max = LaplacianLambdaMax()(data).lambda_max
        outputs = []
        for time_step in range(T):
            outputs.append(torch.unsqueeze(model(encoder_inputs[:,:,:,time_step], edge_index, spatial_attention, lambda_max = lambda_max), -1))
        spatial_gcn = torch.nn.functional.relu(torch.cat(outputs, dim=-1)) # (b,N,F,T) # (b,N,F,T)
    assert spatial_gcn.shape == (batch_size, node_count, nb_chev_filter, T) 