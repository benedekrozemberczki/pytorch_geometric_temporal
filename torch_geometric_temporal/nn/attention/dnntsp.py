import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.utils.to_dense_adj import to_dense_adj
import torch.nn.functional as F

class MaskedSelfAttention(nn.Module):

    def __init__(self, input_dim, output_dim, n_heads=4, attention_aggregate="concat"):
        super(masked_self_attention, self).__init__()
        # aggregate multi-heads by concatenation or mean
        self.attention_aggregate = attention_aggregate

        # the dimension of each head is dq // n_heads
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.n_heads = n_heads

        if attention_aggregate == "concat":
            self.per_head_dim = self.dq = self.dk = self.dv = output_dim // n_heads
        elif attention_aggregate == "mean":
            self.per_head_dim = self.dq = self.dk = self.dv = output_dim
        else:
            raise ValueError(f"wrong value for aggregate {attention_aggregate}")

        self.Wq = nn.Linear(input_dim, n_heads * self.dq, bias=False)
        self.Wk = nn.Linear(input_dim, n_heads * self.dk, bias=False)
        self.Wv = nn.Linear(input_dim, n_heads * self.dv, bias=False)

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: tensor, shape (nodes_num, T_max, features_num)
        Returns:
            output: tensor, shape (nodes_num, T_max, output_dim = features_num)
        """
        seq_length = input_tensor.shape[1]
        # tensor, shape (nodes_num, T_max, n_heads * dim_per_head)
        Q = self.Wq(input_tensor)
        K = self.Wk(input_tensor)
        V = self.Wv(input_tensor)
        # multi_head attention
        # Q, tensor, shape (nodes_num, n_heads, T_max, dim_per_head)
        Q = Q.reshape(input_tensor.shape[0], input_tensor.shape[1], self.n_heads, self.dq).transpose(1, 2)
        # K after transpose, tensor, shape (nodes_num, n_heads, dim_per_head, T_max)
        K = K.reshape(input_tensor.shape[0], input_tensor.shape[1], self.n_heads, self.dk).permute(0, 2, 3, 1)
        # V, tensor, shape (nodes_num, n_heads, T_max, dim_per_head)
        V = V.reshape(input_tensor.shape[0], input_tensor.shape[1], self.n_heads, self.dv).transpose(1, 2)

        # scaled attention_score, tensor, shape (nodes_num, n_heads, T_max, T_max)
        attention_score = Q.matmul(K) / np.sqrt(self.per_head_dim)

        # attention_mask, tensor, shape -> (T_max, T_max)  -inf in the top and right
        attention_mask = torch.zeros(seq_length, seq_length).masked_fill(
            torch.tril(torch.ones(seq_length, seq_length)) == 0, -np.inf).to(input_tensor.device)
        # attention_mask will be broadcast to (nodes_num, n_heads, T_max, T_max)
        attention_score = attention_score + attention_mask
        # (nodes_num, n_heads, T_max, T_max)
        attention_score = torch.softmax(attention_score, dim=-1)

        # multi_result, tensor, shape (nodes_num, n_heads, T_max, dim_per_head)
        multi_head_result = attention_score.matmul(V)
        if self.attention_aggregate == "concat":
            # multi_result, tensor, shape (nodes_num, T_max, n_heads * dim_per_head = output_dim)
            # concat multi-head attention results
            output = multi_head_result.transpose(1, 2).reshape(input_tensor.shape[0],
                                                               seq_length, self.n_heads * self.per_head_dim)
        elif self.attention_aggregate == "mean":
            # multi_result, tensor, shape (nodes_num, T_max, dim_per_head = output_dim)
            # mean multi-head attention results
            output = multi_head_result.transpose(1, 2).mean(dim=2)
        else:
            raise ValueError(f"wrong value for aggregate {self.attention_aggregate}")

        return output


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


class AggregateTemporalNodeFeatures(nn.Module):

    def __init__(self, item_embed_dim):
        """
        :param item_embed_dim: the dimension of input features
        """
        super(aggregate_nodes_temporal_feature, self).__init__()

        self.Wq = nn.Linear(item_embed_dim, 1, bias=False)

    def forward(self, graph, lengths, nodes_output):
        """
        :param graph: batched graphs, with the total number of nodes is `node_num`,
                        including `batch_size` disconnected subgraphs
        :param lengths: tensor, (batch_size, )
        :param nodes_output: the output of self-attention model in time dimension, (n_1+n_2+..., T_max, F)
        :return: aggregated_features, (n_1+n_2+..., F)
        """
        nums_nodes, id = graph.batch_num_nodes(), 0
        aggregated_features = []
        for num_nodes, length in zip(nums_nodes, lengths):
            # get each user's length, tensor, shape, (user_nodes, user_length, item_embed_dim)
            output_node_features = nodes_output[id:id + num_nodes, :length, :]
            # weights for each timestamp, tensor, shape, (user_nodes, 1, user_length)
            # (user_nodes, user_length, 1) transpose to -> (user_nodes, 1, user_length)
            weights = self.Wq(output_node_features).transpose(1, 2)
            # (user_nodes, 1, user_length) matmul (user_nodes, user_length, item_embed_dim)
            # -> (user_nodes, 1, item_embed_dim) squeeze to (user_nodes, item_embed_dim)
            # aggregated_feature, tensor, shape, (user_nodes, item_embed_dim)
            aggregated_feature = weights.matmul(output_node_features).squeeze(dim=1)
            aggregated_features.append(aggregated_feature)
            id += num_nodes
        # (n_1+n_2+..., item_embed_dim)
        aggregated_features = torch.cat(aggregated_features, dim=0)
        return aggregated_features

class WeightedGCNBlock(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: List[int], out_features: int):
        super(weighted_GCN, self).__init__()
        gcns, relus, bns = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        # layers for hidden_size
        input_size = in_features
        for hidden_size in hidden_sizes:
            gcns.append(weighted_graph_conv(input_size, hidden_size))
            relus.append(nn.ReLU())
            bns.append(nn.BatchNorm1d(hidden_size))
            input_size = hidden_size
        # output layer
        gcns.append(weighted_graph_conv(hidden_sizes[-1], out_features))
        relus.append(nn.ReLU())
        bns.append(nn.BatchNorm1d(out_features))
        self.gcns, self.relus, self.bns = gcns, relus, bns

    def forward(self, graph: dgl.DGLGraph, node_features: torch.Tensor, edges_weight: torch.Tensor):
        """
        :param graph: a graph
        :param node_features: shape (n_1+n_2+..., n_features)
               edges_weight: shape (T, n_1^2+n_2^2+...)
        :return:
        """
        h = node_features
        for gcn, relu, bn in zip(self.gcns, self.relus, self.bns):
            # (n_1+n_2+..., T, features)
            h = gcn(graph, h, edges_weight)
            h = bn(h.transpose(1, -1)).transpose(1, -1)
            h = relu(h)
        return h

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
        self.stacked_gcn = WeightedGCNBlock([weighted_GCN(item_embedding_dim,
                                                          [item_embedding_dim],
                                                          item_embedding_dim)])

        self.masked_self_attention = MaskedSelfAttention(input_dim=item_embedding_dim,
                                                         output_dim=item_embedding_dim)

        self.aggregate_nodes_temporal_feature = AggregateTemporalNodeFeatures(item_embed_dim=item_embedding_dim)

        self.global_gated_update = GlobalGatedUpdater(items_total=items_total,
                                                      item_embedding=self.item_embedding)

        self.fully_connected = nn.Linear(item_embedding_dim, 1)

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
        nodes_output = self.stacked_gcn(graph, nodes_feature, edges_weight)
        nodes_output = self.masked_self_attention(nodes_output)
        nodes_output = self.aggregate_nodes_temporal_feature(graph, lengths, nodes_output)
        nodes_output = self.global_gated_updater(graph, nodes, nodes_output)
        output = self.fully_connected(nodes_output).squeeze(dim=-1)
        return output
 
