import torch
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix
import networkx as nx
from torch_geometric_temporal.nn import MTGNN
    
def create_mock_data(number_of_nodes, edge_per_node, in_channels):
    """
    Creating a mock feature matrix and edge index.
    """
    graph = nx.watts_strogatz_graph(number_of_nodes, edge_per_node, 0.5)
    edge_index = torch.LongTensor(np.array([edge for edge in graph.edges()]).T)
    X = torch.FloatTensor(np.random.uniform(-1, 1, (number_of_nodes, in_channels)))
    return X, edge_index

class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


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
total_size = 100
num_batch = int(total_size // batch_size)
x_all = torch.zeros(total_size,seq_in_len,num_nodes,in_dim)
y_all = torch.clip(torch.rand(total_size,seq_out_len,num_nodes,in_dim) * 100 - 20,0,80)
for i in range(total_size):
    for j in range(seq_in_len):
        x, _ = create_mock_data(number_of_nodes=num_nodes, edge_per_node=8, in_channels=in_dim)
        x_all[i,j] = x
data = DataLoaderM(x_all,y_all,batch_size)
scaler = StandardScaler(mean=x_all[..., 0].mean(), std=x_all[..., 0].std())
# define model and optimizer
model = MTGNN(gcn_true, buildA_true, gcn_depth, num_nodes,
              device, predefined_A=predefined_A,
              dropout=dropout, subgraph_size=subgraph_size,
              node_dim=node_dim,
              dilation_exponential=dilation_exponential,
              conv_channels=conv_channels, residual_channels=residual_channels,
              skip_channels=skip_channels, end_channels= end_channels,
              seq_length=seq_in_len, in_dim=in_dim, out_dim=seq_out_len,
              layers=layers, propalpha=propalpha, tanhalpha=tanhalpha, layer_norm_affline=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# begin training
task_level = 1
iter_num = 1
for i in range(epochs):
    for iter, (x, y) in enumerate(data.get_iterator()):
        trainx = torch.Tensor(x).to(device)
        trainx= trainx.transpose(1, 3)
        trainy = torch.Tensor(y).to(device)
        trainy = trainy.transpose(1, 3)
        if iter%step_size2==0:
            perm = np.random.permutation(range(num_nodes))
        num_sub = int(num_nodes/num_split) # number of nodes in each sudgraph
        for j in range(num_split):
            if j != num_split-1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]
            id = torch.tensor(id).to(device) # a permutation of node id
            tx = trainx[:, :, id, :]
            ty = trainy[:, :, id, :]
            model.train()
            optimizer.zero_grad()
            output = model(tx, idx=id)
            output = output.transpose(1,3)
            real = torch.unsqueeze(ty[:,0,:,:],dim=1)
            predict = scaler.inverse_transform(output)
            if iter_num%step_size1==0 and task_level<=seq_out_len:
                task_level +=1
            if cl:
                loss = masked_mae(predict[:, :, :, :task_level], real[:, :, :, :task_level], 0.0)
            else:
                loss = masked_mae(predict, real, 0.0)
            loss.backward()
    
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    
            optimizer.step()
            iter_num += 1


