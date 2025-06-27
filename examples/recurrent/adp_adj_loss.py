import torch
import torch.nn as nn

class AccumulativeGainLoss(nn.Module):
    def __init__(self, value_decay: float = 0.9, penalty_weight: float = 0.1, eps: float = 1e-8):
        """
        Args:
          value_decay: ω，R^2 的衰减因子
          penalty_weight: λ，因子自相关惩罚的权重
        """
        super().__init__()
        self.value_decay = value_decay
        self.penalty_weight = penalty_weight
        self.eps = eps

    def forward(self, preds: torch.Tensor, y_ts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: [N, K] 因子矩阵 F
            y_ts:  [T, N] 未来 T 天的标量收益
        Returns:
            loss = sum_{t=1}^T ω^{t-1} * R²(F, y_t)
                   + λ * sum_{i≠j} corr(F[:,i], F[:,j])²
        """
        N, K = preds.shape
        T = y_ts.shape[0]
        device = preds.device

        # 1. 逐步累加加权 R²
        total_r2 = torch.tensor(0., device=device)
        # 预先计算 (F^T F)^{-1} F^T
        #   F: [N,K] → FtF: [K,K]
        FtF = preds.T @ preds                         # [K,K]
        inv_FtF = torch.inverse(FtF + self.eps * torch.eye(K, device=device))
        pseudo_inv = inv_FtF @ preds.T                # [K,N]

        for t in range(T):
            weight = self.value_decay**t             # ω^{t-1}
            y = y_ts[t]                              # [N]
            # 预测值：F @ pseudo_inv @ y  → [N]
            y_hat = preds @ (pseudo_inv @ y)         # [N]

            # 计算 SSE 和 SST
            ss_res = ((y - y_hat)**2).sum()
            y_bar  = y.mean()
            ss_tot = ((y - y_bar)**2).sum() + self.eps

            r2 = 1 - ss_res / ss_tot
            total_r2 = total_r2 + weight * r2

        # 我们希望 maximize R² → 最后作为要**最小化**的 loss，取负号
        loss_r2 = - total_r2 / T

        # 2. 因子自相关惩罚
        #    计算因子间相关矩阵 corr_mat: [K,K]
        corr_mat = torch.corrcoef(preds.T)
        #    off-diagonal 元素平方和
        eye = torch.eye(K, device=device)
        off_diag = corr_mat[~eye.bool()]
        loss_corr = (off_diag**2).sum()

        # 3. 合并
        loss = loss_r2 + self.penalty_weight * loss_corr
        return loss
