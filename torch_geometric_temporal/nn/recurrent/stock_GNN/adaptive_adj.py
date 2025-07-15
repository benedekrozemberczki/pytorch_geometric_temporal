# models/dyn_graph_module.py
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from torch_geometric.nn import GCNConv, GATConv  # Add GATConv import
from torch_geometric.data import Batch
from torch_geometric_temporal.nn.recurrent.stock_GNN.adp_adj_loss import AccumulativeGainLoss  # 假设你有这个自定义损失函数
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
        metric_compute_frequency: int = 10,  # Compute metrics every N epochs
        weight_decay: float = 1e-4,
        scheduler_config: dict = None,
        gnn_type: str = "gcn",  # Options: "gcn", "gat"
        gat_heads: int = 4,  # Number of attention heads for GAT
        gat_dropout: float = 0.1,  # Dropout for GAT attention
        predict_return: bool = False, # whether predict the final return
        output_factor_dim: int = 32,
    ):
        super().__init__()
        # 1) 用于序列编码的 GRU（或 LSTM/GRUCell）
        self.gru = nn.GRU(
            input_size=node_feat_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
        )
        # 2) 用于图消息传递的 GNN 层
        self.gnn_type = gnn_type.lower()
        self.gat_heads = gat_heads
        self.gat_dropout = gat_dropout
        
        if self.gnn_type == "gat":
            # GAT layers with multi-head attention
            self.gnn1 = GATConv(
                gru_hidden_dim, 
                gnn_hidden_dim // gat_heads,  # Output dim per head
                heads=gat_heads,
                dropout=gat_dropout,
                add_self_loops=add_self_loops,
                concat=True  # Concatenate multi-head outputs
            )
            self.gnn2 = GATConv(
                gnn_hidden_dim,  # Input is concatenated output from gnn1 (heads * out_dim)
                gnn_hidden_dim,  # Final output dimension
                heads=1,  # Single head for final layer
                dropout=gat_dropout,
                add_self_loops=add_self_loops,
                concat=False  # Don't concatenate (only 1 head anyway)
            )
            # Final dimension is what we specify for gnn2
            self.final_gnn_dim = gnn_hidden_dim
        else:
            # Default GCN layers
            self.gnn1 = GCNConv(gru_hidden_dim, gnn_hidden_dim, add_self_loops=add_self_loops)
            self.gnn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim, add_self_loops=add_self_loops)
            self.final_gnn_dim = gnn_hidden_dim
            
        # 3) 最后预测层（添加BatchNorm）
        self.batch_norm = nn.BatchNorm1d(self.final_gnn_dim)
        self.output_factor_dim = output_factor_dim
        self.predictor = nn.Linear(self.final_gnn_dim, self.output_factor_dim)  # 回归示例
        self.predict_return = predict_return
        if self.predict_return:
            self.return_predictor = self.return_predictor = nn.ModuleList([
                nn.ReLU(),  # ReLU activation
                nn.Linear(self.output_factor_dim, 1)  # Linear layer
            ])
        self.k_nn = k_nn
        self.lr = lr
        self.loss_fn = loss_fn
        self.add_self_loops = add_self_loops
        self.metric_compute_frequency = metric_compute_frequency
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config or {}
        
        # Log model configuration
        print(f"🏗️  GNN Architecture: {self.gnn_type.upper()}")
        if self.gnn_type == "gat":
            print(f"   - GAT heads: {self.gat_heads}")
            print(f"   - GAT dropout: {self.gat_dropout}")
            print(f"   - Final GNN dim: {self.final_gnn_dim}")

    def _should_compute_metrics(self) -> bool:
        """Determine if metrics should be computed based on current epoch and frequency."""
        current_epoch = self.current_epoch
        return (current_epoch % self.metric_compute_frequency == 0) or (current_epoch == 0)

    def forward(self, data_input) -> list:
        """
        data_input: 可以是以下两种格式之一:
        1. torch.Tensor: [batch_size, num_nodes, seq_len, node_feat_dim] - 所有图大小相同
        2. list[torch.Tensor]: [graph1_seq, graph2_seq, ...] - 每个图可以有不同大小
           其中每个graph_seq的形状为: [num_nodes_i, seq_len, node_feat_dim]
        
        返回：
        1. 如果输入是tensor: torch.Tensor [batch_size, num_nodes, 1]
        2. 如果输入是list: list[torch.Tensor] [graph1_output, graph2_output, ...]
        """
        if isinstance(data_input, torch.Tensor):
            # 处理固定大小的批量图数据
            return self._forward_batch_tensor(data_input)
        elif isinstance(data_input, list):
            # 处理不同大小的图数据列表
            return self._forward_graph_list(data_input)
        else:
            raise ValueError("Input must be either torch.Tensor or list of torch.Tensor")
    
    def _forward_batch_tensor(self, x_seq: torch.Tensor) -> torch.Tensor:
        """处理固定大小的批量图数据"""
        # print(f"x's shape:{x_seq.shape}")
        # ——— 1. GRU 编码 ——————————————
        if x_seq.dim() == 3:
            # If input is [batch_size, num_nodes, feat_dim], add seq_len dimension
            b, n, f = x_seq.shape
            l = 1
            # x_seq = x_seq.unsqueeze(2)  # [batch_size, num_nodes, 1, feat_dim]
            x_seq = x_seq.unsqueeze(1)  # [batch_size, 1, feat_dim, num_nodes]
        else:
            # b, n, l, f = x_seq.shape
            b, l, f, n = x_seq.shape  # [batch_size, seq_len, feat_dim, num_nodes]
        
        # Reshape for GRU: [seq_len, batch_size * num_nodes, feat_dim]
        # gru_in = x_seq.permute(2, 0, 1, 3).reshape(l, b * n, f)
        gru_in = x_seq.permute(1, 0, 3, 2).reshape(l, b * n, f)  # [seq_len, batch_size * num_nodes, feat_dim]
        gru_out, _ = self.gru(gru_in)    # 输出: [seq_len, batch_size * num_nodes, gru_hidden_dim]

        h = gru_out[-1].view(b, n, -1)  # 取最后一个时间步的输出: [batch_size, num_nodes, gru_hidden_dim]
        # TensorBoard logging: log GRU output stats
        self.log('stats/gru_output_mean', h.mean().item(), on_step=True, on_epoch=False)
        self.log('stats/gru_output_std', h.std().item(), on_step=True, on_epoch=False)
        # ——— 2. 动态构图（相似度 + Top-k） —————
        # 计算节点两两点积
        sim = torch.einsum("bni,bmi->bnm", h, h)  # [b, n, n]
        # TensorBoard logging: log similarity stats
        self.log('stats/sim_mean', sim.mean().item(), on_step=True, on_epoch=False)
        self.log('stats/sim_std', sim.std().item(), on_step=True, on_epoch=False)
        # 使用更稳定的Top-k选择方法
        # 将自环设为很小的值，确保不会被选中
        sim_masked = sim.clone()
        # 将对角线（自环）设为很小的值
        eye_mask = torch.eye(n, device=sim.device).bool().unsqueeze(0).expand(b, -1, -1)
        sim_masked[eye_mask] = -1e9
        
        # 现在直接选择top-k，不需要+1
        topk_vals, topk_idx = sim_masked.topk(self.k_nn, dim=-1, sorted=True)
        
        # 创建源节点索引，确保内存布局连续
        edge_src = torch.arange(n, device=sim.device).unsqueeze(0).unsqueeze(2).expand(b, n, self.k_nn).contiguous()
        edge_dst = topk_idx.contiguous()  # [b, n, k]
        edge_weight = topk_vals.contiguous()  # [b, n, k]

        data_list = []
        for i in range(b):
            # Flatten edge indices and weights for this batch item
            # 使用 contiguous() 确保张量内存布局连续
            src_flat = edge_src[i].contiguous().view(-1)  # [n*k]
            dst_flat = edge_dst[i].contiguous().view(-1)  # [n*k]
            weight_flat = edge_weight[i].contiguous().view(-1)  # [n*k]
            
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
        
        # Add self loops only for GCN (GAT handles them internally)
        if self.gnn_type == "gcn" and self.add_self_loops:
            e_idx, e_w = add_self_loops(
                batch_data.edge_index,
                batch_data.edge_weight,
                fill_value=1.0,       # 或其他你想给自环的权重
                num_nodes=batch_data.num_nodes
            )
        
        # GNN message passing (different for GCN vs GAT)
        if self.gnn_type == "gat":
            # GAT doesn't use edge weights directly in the same way
            out1 = F.relu(self.gnn1(x_all, e_idx))
            out2 = F.relu(self.gnn2(out1, e_idx))
        else:
            # GCN uses edge weights
            out1 = F.relu(self.gnn1(x_all, e_idx, edge_weight=e_w))
            out2 = F.relu(self.gnn2(out1, e_idx, edge_weight=e_w))

        # ——— 4. 预测输出 ——————————————
        # 应用 BatchNorm 然后预测
        out2_normalized = self.batch_norm(out2)  # [N_total, gnn_hidden_dim]
        out = self.predictor(out2_normalized)  # [N_total, 32]
        return_output = out
        if self.predict_return:
            return_output = self.forward_return(out)  # Pass normalized output through return predictor
        final_out = return_output.view(b, n, -1)  # [batch, num_nodes, 32/1]
        # TensorBoard logging: log output stats
        self.log('stats/final_output_mean', final_out.mean().item(), on_step=True, on_epoch=False)
        self.log('stats/final_output_std', final_out.std().item(), on_step=True, on_epoch=False)
        return final_out
    
    def _forward_graph_list(self, graph_list: list) -> list:
        """处理不同大小的图数据列表"""
        results = []
        
        for i, graph_seq in enumerate(graph_list):
            # 每个图独立处理: [num_nodes_i, seq_len, node_feat_dim]
            if graph_seq.dim() == 2:
                # 如果是 [num_nodes, feat_dim]，添加seq_len维度
                # n, f = graph_seq.shape
                f, n = graph_seq.shape  # [num_nodes, feat_dim]
                l = 1
                # graph_seq = graph_seq.unsqueeze(1)  # [num_nodes, 1, feat_dim]
                graph_seq = graph_seq.unsqueeze(0)  # [1, num_nodes, feat_dim]
            else:
                # n, l, f = graph_seq.shape
                l, f, n = graph_seq.shape  # [seq_len, feat_dim, num_nodes]
            
            # TensorBoard logging for individual graphs
            self.log(f'graph_stats/graph_{i}_num_nodes', n, on_step=True, on_epoch=False)
            
            # ——— 1. GRU 编码 ——————————————
            # Reshape for GRU: [seq_len, num_nodes, feat_dim]
            # gru_in = graph_seq.permute(1, 0, 2)  # [seq_len, num_nodes, feat_dim]
            gru_in = graph_seq.permute(0, 2, 1)  # [seq_len, num_nodes, feat_dim]
            gru_out, _ = self.gru(gru_in)    # 输出: [seq_len, num_nodes, gru_hidden_dim]
            
            h = gru_out[-1]  # 取最后一个时间步: [num_nodes, gru_hidden_dim]
            
            # TensorBoard logging: log GRU output stats for this graph
            self.log(f'graph_stats/graph_{i}_gru_output_mean', h.mean().item(), on_step=True, on_epoch=False)
            self.log(f'graph_stats/graph_{i}_gru_output_std', h.std().item(), on_step=True, on_epoch=False)
            
            # ——— 2. 动态构图（相似度 + Top-k） —————
            # 计算节点两两点积 [num_nodes, num_nodes]
            sim = torch.einsum("ni,mi->nm", h, h)
            
            # TensorBoard logging: log similarity stats for this graph
            self.log(f'graph_stats/graph_{i}_sim_mean', sim.mean().item(), on_step=True, on_epoch=False)
            self.log(f'graph_stats/graph_{i}_sim_std', sim.std().item(), on_step=True, on_epoch=False)
            
            # 确保k_nn不超过节点数-1
            k = min(self.k_nn, n - 1)
            if k <= 0:
                # 如果图太小，直接用线性层处理
                h_normalized = self.batch_norm(h)  # 应用BatchNorm
                out = self.predictor(h_normalized)  # [num_nodes, 32]
                results.append(out)
                continue
            
            # Top-k选择
            topk_vals, topk_idx = sim.topk(k + 1, dim=-1)  # 包含自己
            
            # 去掉自环
            node_idx = torch.arange(n, device=sim.device)[:, None]  # [n, 1]
            mask = topk_idx != node_idx  # [n, k+1]
            
            # 提取前k个邻居（排除自己）
            edge_weights = []
            edge_sources = []
            edge_targets = []
            
            for node in range(n):
                valid_neighbors = topk_idx[node][mask[node]][:k]  # 取前k个
                valid_weights = topk_vals[node][mask[node]][:k]
                
                # 添加边
                edge_sources.extend([node] * len(valid_neighbors))
                edge_targets.extend(valid_neighbors.tolist())
                edge_weights.extend(valid_weights.tolist())
            
            if len(edge_sources) == 0:
                # 如果没有边，直接用线性层
                h_normalized = self.batch_norm(h)  # 应用BatchNorm
                out = self.predictor(h_normalized)
                results.append(out)
                continue
            
            # 构建图数据
            edge_index = torch.tensor([edge_sources, edge_targets], 
                                    dtype=torch.long, device=h.device)
            edge_weight = torch.tensor(edge_weights, dtype=torch.float, device=h.device)
            
            # 添加自环（如果需要且使用GCN）
            if self.gnn_type == "gcn" and self.add_self_loops:
                edge_index, edge_weight = add_self_loops(
                    edge_index, edge_weight,
                    fill_value=1.0, num_nodes=n
                )
            
            # ——— 3. GNN 消息传递 ——————————————
            if self.gnn_type == "gat":
                # GAT doesn't use edge weights directly
                out1 = F.relu(self.gnn1(h, edge_index))
                out2 = F.relu(self.gnn2(out1, edge_index))
            else:
                # GCN uses edge weights
                out1 = F.relu(self.gnn1(h, edge_index, edge_weight=edge_weight))
                out2 = F.relu(self.gnn2(out1, edge_index, edge_weight=edge_weight))
            
            # ——— 4. 预测输出 ——————————————
            # 应用 BatchNorm 然后预测
            out2_normalized = self.batch_norm(out2)  # [num_nodes, gnn_hidden_dim]
            out = self.predictor(out2_normalized)  # [num_nodes, 32]
            
            # TensorBoard logging: log output stats for this graph
            self.log(f'graph_stats/graph_{i}_output_mean', out.mean().item(), on_step=True, on_epoch=False)
            self.log(f'graph_stats/graph_{i}_output_std', out.std().item(), on_step=True, on_epoch=False)
            
            results.append(out)
        
        return results


    def forward_return(self, x):
        for layer in self.return_predictor:
            x = layer(x)
        return x


    def training_step(self, batch, batch_idx):
        # batch: 来自你的 DataLoader，格式可以是 (x_t, y_t)
        x_t, y_t = batch
        y_pred = self(x_t)
        
        # Determine if we should compute metrics this epoch
        compute_metrics = self._should_compute_metrics()
        
        # 处理不同的输出格式
        if isinstance(y_pred, list):
            # 变长图的情况
            total_loss = 0
            total_nodes = 0
            for i, (pred, target) in enumerate(zip(y_pred, y_t)):
                if isinstance(target, torch.Tensor):
                    loss_i = self.loss_fn(pred.squeeze(-1), target.float(), compute_metrics=compute_metrics)
                    total_loss += loss_i * pred.size(0)  # 按节点数加权
                    total_nodes += pred.size(0)
            loss = total_loss / total_nodes if total_nodes > 0 else total_loss
        else:
            # 固定大小图的情况
            loss = self.loss_fn(y_pred.squeeze(-1), y_t.float(), compute_metrics=compute_metrics)
            
            # 只在计算指标的epoch记录RankIC和ICIR
            if compute_metrics and hasattr(loss, 'rank_ic_info'):
                rank_ic_info = loss.rank_ic_info
                for metric_name, metric_value in rank_ic_info.items():
                    if isinstance(metric_value, (int, float)) and metric_value != 0.0:
                        self.log(f'train_{metric_name}', metric_value, 
                                 on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # TensorBoard logging: log training loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x_t, y_t = batch
        y_pred = self(x_t)
        
        # Determine if we should compute metrics this epoch
        compute_metrics = self._should_compute_metrics()
        
        # 处理不同的输出格式
        if isinstance(y_pred, list):
            # 变长图的情况
            total_loss = 0
            total_nodes = 0
            for i, (pred, target) in enumerate(zip(y_pred, y_t)):
                if isinstance(target, torch.Tensor):
                    loss_i = self.loss_fn(pred.squeeze(-1), target.float(), compute_metrics=compute_metrics)
                    total_loss += loss_i * pred.size(0)  # 按节点数加权
                    total_nodes += pred.size(0)
            loss = total_loss / total_nodes if total_nodes > 0 else total_loss
        else:
            # 固定大小图的情况
            loss = self.loss_fn(y_pred.squeeze(-1), y_t.float(), compute_metrics=compute_metrics)
            
            # 只在计算指标的epoch记录RankIC和ICIR
            if compute_metrics and hasattr(loss, 'rank_ic_info'):
                rank_ic_info = loss.rank_ic_info
                for metric_name, metric_value in rank_ic_info.items():
                    if isinstance(metric_value, (int, float)) and metric_value != 0.0:
                        self.log(f'val_{metric_name}', metric_value, 
                                 on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # TensorBoard logging: log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        
        return loss


    def test_step(self, batch, batch_idx):
        x_t, y_t = batch
        y_pred = self(x_t)
        
        # Always compute metrics during testing for final evaluation
        compute_metrics = True
        
        if isinstance(y_pred, list):
            total_loss = 0.0
            total_nodes = 0
            for pred, target in zip(y_pred, y_t):
                loss_i = self.loss_fn(pred.squeeze(-1), target.float(), compute_metrics=compute_metrics)
                total_loss += loss_i * pred.size(0)
                total_nodes += pred.size(0)
            loss = total_loss / total_nodes if total_nodes>0 else total_loss
        else:
            loss = self.loss_fn(y_pred.squeeze(-1), y_t.float(), compute_metrics=compute_metrics)
            
            # Log all test metrics
            if hasattr(loss, 'rank_ic_info'):
                rank_ic_info = loss.rank_ic_info
                for metric_name, metric_value in rank_ic_info.items():
                    if isinstance(metric_value, (int, float)) and metric_value != 0.0:
                        self.log(f'test_{metric_name}', metric_value, 
                                 on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                        print(f"Final test {metric_name}: {metric_value:.6f}")
                        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar = True, logger=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Check if scheduler is configured
        scheduler_type = self.scheduler_config.get('type', None)
        
        if scheduler_type is None:
            # No scheduler
            return optimizer
        
        elif scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',  # Monitor validation loss (minimize)
                factor=self.scheduler_config.get('factor', 0.5),
                patience=self.scheduler_config.get('patience', 5),
                min_lr=self.scheduler_config.get('min_lr', 1e-6),
                verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',  # Monitor validation loss
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        
        elif scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.get('T_max', 100),
                eta_min=self.scheduler_config.get('min_lr', 1e-6)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_config.get('step_size', 30),
                gamma=self.scheduler_config.get('gamma', 0.1)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        
        elif scheduler_type == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.scheduler_config.get('gamma', 0.95)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        
        else:
            # Unknown scheduler type, return optimizer only
            print(f"Warning: Unknown scheduler type '{scheduler_type}', using no scheduler")
            return optimizer

