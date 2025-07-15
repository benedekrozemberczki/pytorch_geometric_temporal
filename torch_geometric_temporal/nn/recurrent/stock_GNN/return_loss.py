import torch
import torch.nn as nn
from typing import Any
import scipy.stats as stats
import numpy as np  

class ReturnLoss(nn.Module):
    def __init__(self, value_decay: float = 0.9, penalty_weight: float = 0.1, eps: float = 1e-8, importance_weights: Any = [1.0, 0.0, 0.0]):
        super().__init__()
        self.value_decay = value_decay
        self.penalty_weight = penalty_weight
        self.eps = eps
        # 将列表或 ListConfig 转为 Tensor
        if not isinstance(importance_weights, torch.Tensor):
            self.importance_weights = torch.tensor(importance_weights, dtype=torch.float32)
        else:
            self.importance_weights = importance_weights
            
        # 用于存储历史RankIC值，计算ICIR
        self.rank_ic_history = []
        self.abs_rank_ic_history = []


    def forward(self, preds: torch.Tensor, y_ts: torch.Tensor, compute_metrics: bool = False) -> torch.Tensor:  
        """
        Args:
            preds: [B, N, 1] 模型输出收益率（每个 batch 一个图）
            y_ts:  [B, T, N, D] 每个 batch 的未来 T 天收益（D 维）
            importance: [D] 各收益维度的重要性权重

        Returns:
            scalar loss
        """
        B, N = preds.shape
        _, T, _, D = y_ts.shape
        device = preds.device
        
        # 确保 importance 张量在正确的设备上
        self.importance_weights = self.importance_weights.to(device)

        # 只计算 T 维度上第一个和 D 维度上第一个的 MSE loss
        y_ts_selected = y_ts[:, 0, :, 0].squeeze(-1)  # 选择 T 维度第一个和 D 维度第一个
        preds_selected = preds.squeeze(-1)  # 去掉最后一个维度 [B, N]
        assert preds_selected.shape == y_ts_selected.shape, f"Shape mismatch: {preds_selected.shape} vs {y_ts_selected.shape}"
        loss = torch.mean((preds_selected - y_ts_selected) ** 2)  # MSE loss

        # 将 RankIC 和 ICIR 信息作为属性附加到 loss tensor 上
        # 这样外部可以直接访问而不需要重新计算
        if compute_metrics:
            loss.rank_ic_info = {
                # 可以在这里添加 RankIC 和 ICIR 的计算逻辑
            }
        
        return loss
