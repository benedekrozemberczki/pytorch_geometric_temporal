import math
import torch
import numpy as np
import torch.nn as nn
from typing import List
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class MaskedSelfAttention(nn.Module):

    def __init__(self, input_dim, output_dim, n_heads, attention_aggregate="concat"):
        super(MaskedSelfAttention, self).__init__()

        self.attention_aggregate = attention_aggregate
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
            input_tensor:
        Returns:
            output:
        """
        seq_length = input_tensor.shape[1]

        Q = self.Wq(input_tensor)
        K = self.Wk(input_tensor)
        V = self.Wv(input_tensor)

        Q = Q.reshape(input_tensor.shape[0], input_tensor.shape[1], self.n_heads, self.dq).transpose(1, 2)
        K = K.reshape(input_tensor.shape[0], input_tensor.shape[1], self.n_heads, self.dk).permute(0, 2, 3, 1)
        V = V.reshape(input_tensor.shape[0], input_tensor.shape[1], self.n_heads, self.dv).transpose(1, 2)

        attention_score = Q.matmul(K) / np.sqrt(self.per_head_dim)

        attention_mask = torch.zeros(seq_length, seq_length).masked_fill(
            torch.tril(torch.ones(seq_length, seq_length)) == 0, -np.inf).to(input_tensor.device)

        attention_score = attention_score + attention_mask

        attention_score = torch.softmax(attention_score, dim=-1)

        multi_head_result = attention_score.matmul(V)

        if self.attention_aggregate == "concat":
            output = multi_head_result.transpose(1, 2).reshape(input_tensor.shape[0],
                                                               seq_length, self.n_heads * self.per_head_dim)
        elif self.attention_aggregate == "mean":
            output = multi_head_result.transpose(1, 2).mean(dim=2)
        else:
            raise ValueError(f"wrong value for aggregate {self.attention_aggregate}")

        return output


class GlobalGatedUpdater(nn.Module):

    def __init__(self, items_total, item_embedding):
        super(GlobalGatedUpdater, self).__init__()
        self.items_total = items_total
        self.item_embedding = item_embedding
        self.alpha = nn.Parameter(torch.rand(items_total, 1), requires_grad=True)

    def forward(self, nodes, nodes_output):
        """
        :param nodes:
        :param nodes_output:
        :return:
        """
        items_embedding = self.item_embedding(torch.tensor([i for i in range(self.items_total)]).to(nodes.device))
        alpha = torch.sigmoid(self.alpha)
        embed = (1 - alpha) * items_embedding.clone() + alpha * nodes_output
        return embed


class AggregateTemporalNodeFeatures(nn.Module):

    def __init__(self, item_embed_dim):
        """
        :param item_embed_dim:
        """
        super(AggregateTemporalNodeFeatures, self).__init__()

        self.Wq = nn.Linear(item_embed_dim, 1, bias=False)

    def forward(self, lengths, nodes_output):
        """
        :param lengths:
        :param nodes_output:
        :return:
        """
        aggregated_features = []
        for length in lengths:
            output_node_features = nodes_output[:, :length, :]
            weights = self.Wq(output_node_features).transpose(1, 2)
            aggregated_feature = weights.matmul(output_node_features).squeeze(dim=1)
            aggregated_features.append(aggregated_feature)
        aggregated_features = torch.cat(aggregated_features, dim=0)
        return aggregated_features

class WeightedGCNBlock(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: List[int], out_features: int):
        super(WeightedGCNBlock, self).__init__()
        gcns, relus, bns = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        input_size = in_features
        for hidden_size in hidden_sizes:
            gcns.append(GCNConv(input_size, hidden_size))
            relus.append(nn.ReLU())
            bns.append(nn.BatchNorm1d(hidden_size))
            input_size = hidden_size
        gcns.append(GCNConv(hidden_sizes[-1], out_features))
        relus.append(nn.ReLU())
        bns.append(nn.BatchNorm1d(out_features))
        self.gcns, self.relus, self.bns = gcns, relus, bns

    def forward(self, node_features: torch.FloatTensor, edge_index: torch.LongTensor, edges_weight: torch.LongTensor):
        """
        :param graph:
        :param node_features:
        :param edges_weight:
        :return:
        """
        h = node_features
        for gcn, relu, bn in zip(self.gcns, self.relus, self.bns):
            h = gcn(h, edge_index, edges_weight)
            h = bn(h.transpose(1, -1)).transpose(1, -1)
            h = relu(h)
        return h

class DNNTSP(nn.Module):


    def __init__(self, items_total, item_embedding_dim, n_heads):
        """
        :param items_total:
        :param item_embedding_dim:
        :param n_heads:
        :param attention_aggregate:
        """
        super(DNNTSP, self).__init__()

        self.item_embedding = nn.Embedding(items_total, item_embedding_dim)

        self.item_embedding_dim = item_embedding_dim
        self.items_total = items_total
        self.stacked_gcn = WeightedGCNBlock(item_embedding_dim, [item_embedding_dim], item_embedding_dim)

        self.masked_self_attention = MaskedSelfAttention(input_dim=item_embedding_dim,
                                                         output_dim=item_embedding_dim,
                                                         n_heads=n_heads)

        self.aggregate_nodes_temporal_feature = AggregateTemporalNodeFeatures(item_embed_dim=item_embedding_dim)

        self.global_gated_update = GlobalGatedUpdater(items_total=items_total,
                                                      item_embedding=self.item_embedding)

        self.fully_connected = nn.Linear(item_embedding_dim, 1)

    def forward(self, node_features: torch.Tensor, edge_index: torch.LongTensor, edges_weight: torch.FloatTensor,
                lengths: torch.Tensor, nodes: torch.Tensor, users_frequency: torch.Tensor):
        """
        :param graph:
        :param nodes_feature:
        :param edges_weight:
        :param lengths:
        :param nodes:
        :param users_frequency:
        :return:
        """
        nodes_output = self.stacked_gcn(nodes_features, edges_weight)
        nodes_output = self.masked_self_attention(nodes_output)
        nodes_output = self.aggregate_nodes_temporal_feature(graph, lengths, nodes_output)
        nodes_output = self.global_gated_updater(graph, nodes, nodes_output)
        output = self.fully_connected(nodes_output).squeeze(dim=-1)
        return output
 
