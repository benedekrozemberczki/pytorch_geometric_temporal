# models/dyn_graph_module.py
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from torch_geometric.nn import GCNConv  # 或者 GATConv, GraphConv…
from torch_geometric.data import Batch
from adp_adj_loss import AccumulativeGainLoss  # 假设你有这个自定义损失函数

class DynamicGraphLightning(pl.LightningModule):
    def __init__(
        self,
        node_feat_dim: int,
        gru_hidden_dim: int = 64,
        gnn_hidden_dim: int = 64,
        k_nn: int = 8,
        lr: float = 1e-3,
        loss_fn: nn.Module = AccumulativeGainLoss(value_decay=0.9, penalty_weight=0.1, eps=1e-8),
    ):
        super().__init__()
        # 1) 用于序列编码的 GRU（或 LSTM/GRUCell）
        self.gru = nn.GRU(
            input_size=node_feat_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
        )
        # 2) 用于图消息传递的 GNN 层
        self.gnn1 = GCNConv(gru_hidden_dim, gnn_hidden_dim)
        self.gnn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
        # 3) 最后预测层
        self.predictor = nn.Linear(gnn_hidden_dim, 1)  # 回归示例

        self.k_nn = k_nn
        self.lr = lr
        self.loss_fn = loss_fn

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: [batch_size, num_nodes, node_feat_dim]
        返回：
          out: [batch_size, num_nodes, 1]
        """
        # ——— 1. GRU 编码 ——————————————
        # 这里我们把 batch_size 当作序列长度来看，或者自行调整
        # 假设 x_seq 已经是 [batch_size, seq_len=1, num_nodes, feat]
        # 你也可以在 Dataset 把 history window pack 到 feat 维度里
        b, n, f = x_seq.shape
        # 为了简单，把 seq_len=1：
        gru_in = x_seq.view(b, 1, n * f)  # 例子：flatten 后当作序列
        gru_out, _ = self.gru(gru_in)    # [b, 1, hidden]
        h = gru_out.squeeze(1)           # [b, hidden]

        # 把 GRU 隐状态再 split 回节点维度
        h = h.view(b, n, -1)             # [b, num_nodes, gru_hidden_dim]

        # ——— 2. 动态构图（相似度 + Top-k） —————
        # 计算节点两两点积
        sim = torch.einsum("bni,bmi->bnm", h, h)  # [b, n, n]
        # Top-k
        topk_vals, topk_idx = sim.topk(self.k_nn + 1, dim=-1)  # 包含自己
        # 去掉自环
        node_idx = torch.arange(n, device=sim.device)[None, :, None]
        mask = topk_idx != node_idx
        edge_weight = topk_vals[mask].view(b, n * self.k_nn)
        edge_dst    = topk_idx[mask].view(b, n * self.k_nn)
        edge_src    = node_idx.expand(b, n, self.k_nn)[mask].view(b, n * self.k_nn)

        # 把 batch 展平成一个大图: 假设 Lightning 的 batch_size=1，或自行处理
        edge_index = torch.stack([edge_src, edge_dst], dim=0)  # [2, n*k] or [2, b*n*k]

        # ——— 3. GNN 消息传递 ——————————————
        # 把节点特征 flatten 成 [N_total, feat]
        x = h.view(-1, h.size(-1))
        e_idx = edge_index.view(2, -1)
        e_w   = edge_weight.view(-1)

        x = F.relu(self.gnn1(x, e_idx, edge_weight=e_w))
        x = F.relu(self.gnn2(x, e_idx, edge_weight=e_w))

        # ——— 4. 预测输出 ——————————————
        out = self.predictor(x)  # [N_total, 1]
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

