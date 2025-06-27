# models/dyn_graph_module.py
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from torch_geometric.nn import GCNConv  # 或者 GATConv, GraphConv…
from torch_geometric.data import Batch
from torch_geometric_temporal.nn.recurrent.adp_adj_loss import AccumulativeGainLoss  # 假设你有这个自定义损失函数
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops

class DynamicGraphLightning(pl.LightningModule):
    def __init__(
        self,
        node_feat_dim: int,
        gru_hidden_dim: int = 64,
        gnn_hidden_dim: int = 64,
        k_nn: int = 8,
        lr: float = 1e-3,
        loss_fn: nn.Module = AccumulativeGainLoss(value_decay=0.9, penalty_weight=0.1, eps=1e-8),
        add_self_loops: bool = True,
    ):
        super().__init__()
        # 1) 用于序列编码的 GRU（或 LSTM/GRUCell）
        self.gru = nn.GRU(
            input_size=node_feat_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
        )
        # 2) 用于图消息传递的 GNN 层
        self.gnn1 = GCNConv(gru_hidden_dim, gnn_hidden_dim, add_self_loops=add_self_loops)
        self.gnn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim, add_self_loops=add_self_loops)
        # 3) 最后预测层
        self.predictor = nn.Linear(gnn_hidden_dim, 1)  # 回归示例

        self.k_nn = k_nn
        self.lr = lr
        self.loss_fn = loss_fn
        self.add_self_loops = add_self_loops

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: [batch_size, num_nodes, seq_len, node_feat_dim] or [batch_size, num_nodes, node_feat_dim]
        返回：
          out: [batch_size, num_nodes, 1]
        """
        # ——— 1. GRU 编码 ——————————————
        if x_seq.dim() == 3:
            # If input is [batch_size, num_nodes, feat_dim], add seq_len dimension
            b, n, f = x_seq.shape
            l = 1
            x_seq = x_seq.unsqueeze(2)  # [batch_size, num_nodes, 1, feat_dim]
        else:
            b, n, l, f = x_seq.shape
        
        # Reshape for GRU: [seq_len, batch_size * num_nodes, feat_dim]
        gru_in = x_seq.permute(2, 0, 1, 3).reshape(l, b * n, f)
        gru_out, _ = self.gru(gru_in)    # 输出: [seq_len, batch_size * num_nodes, gru_hidden_dim]

        h = gru_out[-1].view(b, n, -1)  # 取最后一个时间步的输出: [batch_size, num_nodes, gru_hidden_dim]
        print(f"GRU output shape: {h.shape}")
        # ——— 2. 动态构图（相似度 + Top-k） —————
        # 计算节点两两点积
        sim = torch.einsum("bni,bmi->bnm", h, h)  # [b, n, n]
        print(f"Similarity matrix shape: {sim.shape}")
        # Top-k
        topk_vals, topk_idx = sim.topk(self.k_nn + 1, dim=-1)  # 包含自己
        print(f"Top-k values shape: {topk_vals.shape}, Top-k indices shape: {topk_idx.shape}")
        # 去掉自环
        node_idx = torch.arange(n, device=sim.device)[None, :, None]
        node_idx_k1 = node_idx.view(1, n, 1).expand(b, n, self.k_nn+1)  # [b,n,k+1]

        # 3) mask 去掉自循环 (i != i)
        mask = topk_idx != node_idx_k1                         # [b,n,k+1]

        # 4) 用同一个 mask 索引，留下 k 条真正的邻居边
        edge_weight = topk_vals[mask].view(b, n, self.k_nn)    # [b, n, k]
        edge_dst    = topk_idx[mask].view(b, n, self.k_nn)     # [b, n, k]
        edge_src    = node_idx_k1[mask].view(b, n, self.k_nn)  # [b, n, k]

        data_list = []
        for i in range(b):
            # Flatten edge indices and weights for this batch item
            src_flat = edge_src[i].view(-1)  # [n*k]
            dst_flat = edge_dst[i].view(-1)  # [n*k]
            weight_flat = edge_weight[i].view(-1)  # [n*k]
            
            data = Data(
                x=h[i],  # [n, feat_dim]
                edge_index=torch.stack([src_flat, dst_flat], dim=0),  # [2, n*k]
                edge_weight=weight_flat  # [n*k]
            )
            data_list.append(data)
        # # 把 batch 展平成一个大图: 假设 Lightning 的 batch_size=1，或自行处理
        # edge_index = torch.stack([edge_src, edge_dst], dim=0)  # [2, n*k] or [2, b*n*k]

        # # ——— 3. GNN 消息传递 ——————————————
        # # 把节点特征 flatten 成 [N_total, feat]
        # x = h.view(-1, h.size(-1))
        # e_idx = edge_index.view(2, -1)
        # e_w   = edge_weight.view(-1)
        batch_data = Batch.from_data_list(data_list)
        x_all    = batch_data.x            # [b*n, feat_dim]
        e_idx    = batch_data.edge_index   # [2, b*n*k]
        e_w      = batch_data.edge_weight  # [b*n*k]
        print(f"Batch data shape: {x_all.shape}, Edge index shape: {e_idx.shape}, Edge weight shape: {e_w.shape}")
        if(self.add_self_loops):
            e_idx, e_w = add_self_loops(
                batch_data.edge_index,
                batch_data.edge_weight,
                fill_value=1.0,       # 或其他你想给自环的权重
                num_nodes=batch_data.num_nodes
            )
        # 第一层
        out1 = F.relu(self.gnn1(x_all, e_idx, edge_weight=e_w))
        # 第二层
        out2 = F.relu(self.gnn2(out1, e_idx, edge_weight=e_w)) #[b*n, feat_dim]

        # ——— 4. 预测输出 ——————————————
        out = self.predictor(out2)  # [N_total, 1]
        return out.view(b, n, -1)  # [batch, num_nodes, 1]

    def training_step(self, batch, batch_idx):
        # batch: 来自你的 DataLoader，格式可以是 (x_t, y_t)
        x_t, y_t = batch
        y_pred = self(x_t)
        # loss = F.mse_loss(y_pred.squeeze(-1), y_t.float())
        loss = self.loss_fn(y_pred.squeeze(-1), y_t.float())
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_t, y_t = batch
        y_pred = self(x_t)
        # loss = F.mse_loss(y_pred.squeeze(-1), y_t.float())
        loss = self.loss_fn(y_pred.squeeze(-1), y_t.float())
        self.log("val/loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

