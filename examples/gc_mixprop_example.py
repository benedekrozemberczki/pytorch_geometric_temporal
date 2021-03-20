import torch
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix
import networkx as nx
from torch_geometric_temporal.nn import mixprop, graph_constructor
    
def create_mock_data(number_of_nodes, edge_per_node, in_channels):
    """
    Creating a mock feature matrix and edge index.
    """
    graph = nx.watts_strogatz_graph(number_of_nodes, edge_per_node, 0.5)
    edge_index = torch.LongTensor(np.array([edge for edge in graph.edges()]).T)
    X = torch.FloatTensor(np.random.uniform(-1, 1, (number_of_nodes, in_channels)))
    return X, edge_index


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
torch.set_num_threads(3)
x, edge_index = create_mock_data(number_of_nodes=num_nodes, edge_per_node=8, in_channels=in_dim)
mock_adj = to_scipy_sparse_matrix(edge_index)
predefined_A = torch.tensor(mock_adj.toarray()).to(device)
total_size = batch_size
num_batch = int(total_size // batch_size)
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
x_tmp = start_conv(x_all[:batch_size].transpose(1,3))
model = mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha)
mixprop_output = model(x_tmp,adp)


