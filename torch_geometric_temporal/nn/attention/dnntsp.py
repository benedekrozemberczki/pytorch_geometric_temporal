import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.utils.to_dense_adj import to_dense_adj
import torch.nn.functional as F


class GlobalGatedUpdater(nn.Module):

    def __init__(self, items_total, item_embedding):
        super(global_gated_update, self).__init__()
        self.items_total = items_total
        self.item_embedding = item_embedding

        # alpha -> the weight for updating
        self.alpha = nn.Parameter(torch.rand(items_total, 1), requires_grad=True)

    def forward(self, graph, nodes, nodes_output):
        """
        :param graph: batched graphs, with the total number of nodes is `node_num`,
                        including `batch_size` disconnected subgraphs
        :param nodes: tensor (n_1+n_2+..., )
        :param nodes_output: the output of self-attention model in time dimension, (n_1+n_2+..., F)
        :return:
        """
        nums_nodes, id = graph.batch_num_nodes(), 0
        items_embedding = self.item_embedding(torch.tensor([i for i in range(self.items_total)]).to(nodes.device))
        batch_embedding = []
        for num_nodes in nums_nodes:
            # tensor, shape, (user_nodes, item_embed_dim)
            output_node_features = nodes_output[id:id + num_nodes, :]
            # get each user's nodes
            output_nodes = nodes[id: id + num_nodes]
            # beta, tensor, (items_total, 1), indicator vector, appear item -> 1, not appear -> 0
            beta = torch.zeros(self.items_total, 1).to(nodes.device)
            beta[output_nodes] = 1
            # update global embedding by gated mechanism
            # broadcast (items_total, 1) * (items_total, item_embed_dim) -> (items_total, item_embed_dim)
            embed = (1 - beta * self.alpha) * items_embedding.clone()
            # appear items: (1 - self.alpha) * origin + self.alpha * update, not appear items: origin
            embed[output_nodes, :] = embed[output_nodes, :] + self.alpha[output_nodes] * output_node_features
            batch_embedding.append(embed)
            id += num_nodes
        # (B, items_total, item_embed_dim)
        batch_embedding = torch.stack(batch_embedding)
        return batch_embedding
 
class DNNTSP(nn.Module):


    def __init__(self, items_total, item_embedding_dim):
        """
        :param items_total: int
        :param item_embedding_dim: int
        :param n_heads: int
        :param attention_aggregate: sre
        """
        super(temporal_set_prediction, self).__init__()

        self.item_embedding = nn.Embedding(items_total, item_embedding_dim)

        self.item_embedding_dim = item_embedding_dim
        self.items_total = items_total
        self.stacked_gcn = stacked_weighted_GCN_blocks([weighted_GCN(item_embedding_dim,
                                                                     [item_embedding_dim],
                                                                     item_embedding_dim)])

        self.masked_self_attention = masked_self_attention(input_dim=item_embedding_dim,
                                                           output_dim=item_embedding_dim)

        self.aggregate_nodes_temporal_feature = aggregate_nodes_temporal_feature(item_embed_dim=item_embedding_dim)

        self.global_gated_update = GlobalGatedUpdater(items_total=items_total,
                                                      item_embedding=self.item_embedding)

        self.fc_output = nn.Linear(item_embedding_dim, 1)

    def forward(self, graph: dgl.DGLGraph, nodes_feature: torch.Tensor, edges_weight: torch.Tensor,
                lengths: torch.Tensor, nodes: torch.Tensor, users_frequency: torch.Tensor):
        """
        :param graph: batched graphs, with the total number of nodes is `node_num`,
                        including `batch_size` disconnected subgraphs
        :param nodes_feature:  [n_1+n_2+..., F]
        :param edges_weight: [T_max, n_1^2+n_2^2+...]
        :param lengths: [batch_size, ]
        :param nodes: [n_1+n_2+..., ]
        :param users_frequency: (batch, items_total), for frequency calculation
        :return:
        """
        # perform weighted gcn on dynamic graphs (n_1+n_2+..., T_max, item_embed_dim)
        nodes_output = self.stacked_gcn(graph, nodes_feature, edges_weight)

        # self-attention in time dimension, (n_1+n_2+..., T_max,  item_embed_dim)
        nodes_output = self.masked_self_attention(nodes_output)
        # aggregate node features in temporal dimension, (n_1+n_2+..., item_embed_dim)
        nodes_output = self.aggregate_nodes_temporal_feature(graph, lengths, nodes_output)

        # (batch_size, items_total, item_embed_dim)
        nodes_output = self.global_gated_update(graph, nodes, nodes_output)

        # (batch_size, items_total)
        output = self.fc_output(nodes_output).squeeze(dim=-1)

        return output
 
