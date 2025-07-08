import torch
import torch.nn as nn

class AccumulativeGainLoss(nn.Module):
    def __init__(self, value_decay: float = 0.9, penalty_weight: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.value_decay = value_decay
        self.penalty_weight = penalty_weight
        self.eps = eps

    def forward(self, preds: torch.Tensor, y_ts: torch.Tensor, importance: torch.Tensor = torch.tensor([1.0, 0.0, 0.0])) -> torch.Tensor:
        """
        Args:
            preds: [B, N, K] 模型输出因子（每个 batch 一个图）
            y_ts:  [B, T, N, D] 每个 batch 的未来 T 天收益（D 维）
            importance: [D] 各收益维度的重要性权重

        Returns:
            scalar loss
        """
        B, N, K = preds.shape
        _, T, _, D = y_ts.shape
        device = preds.device
        
        # 确保 importance 张量在正确的设备上
        importance = importance.to(device)

        total_loss_r2 = 0.0
        total_loss_corr = 0.0

        for b in range(B):
            F_b = preds[b]          # [N, K]
            y_b = y_ts[b]           # [T, N, D]

            # === 计算 pseudo-inverse: (F^T F)^(-1) F^T ===
            FtF = F_b.T @ F_b
            inv_FtF = torch.inverse(FtF + self.eps * torch.eye(K, device=device))
            pseudo_inv = inv_FtF @ F_b.T     # [K, N]

            total_r2 = 0.0
            for t in range(T):
                weight_t = self.value_decay ** t
                y_t = y_b[t]       # [N, D]
                y_hat = F_b @ (pseudo_inv @ y_t)  # [N, D]

                ss_res = ((y_t - y_hat) ** 2).sum(dim=0)  # [D]
                y_mean = y_t.mean(dim=0)
                ss_tot = ((y_t - y_mean) ** 2).sum(dim=0) + self.eps  # [D]

                r2_d = 1 - ss_res / ss_tot       # [D]
                weighted_r2 = (r2_d * importance).sum()  # scalar

                total_r2 += weight_t * weighted_r2

            # 最小化负的 R²
            loss_r2 = - total_r2 / T
            total_loss_r2 += loss_r2

            # === 计算信息冗余惩罚 corr(F_b.T) ===
            corr_mat = torch.corrcoef(F_b.T)         # [K, K]
            eye = torch.eye(K, device=device)
            off_diag = corr_mat[~eye.bool()]         # [K*K - K]
            loss_corr = (off_diag ** 2).sum()
            total_loss_corr += loss_corr

        # 求所有 batch 平均
        loss = total_loss_r2 / B + self.penalty_weight * (total_loss_corr / B)
        return loss
