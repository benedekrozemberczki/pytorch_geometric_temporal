from dnntsp import DNNTSP
import torch
import networkx as nx
import numpy as np

model = DNNTSP(items_total=100, item_embedding_dim=16, n_heads=4)

g = nx.watts_strogatz_graph(100, 10, 0.4)

edges = torch.LongTensor(np.array([[edge[0], edge[1]] for edge in g.edges()])).T

edge_weight = torch.FloatTensor(np.random.uniform(0, 1 ,(500, )))

node_features = torch.FloatTensor(np.random.uniform(0, 1, (100, 16)))

z = model(node_features, edges, edge_weight, [0])

print(z.shape)
