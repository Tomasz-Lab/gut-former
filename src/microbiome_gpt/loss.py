import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

class CompositeLoss(nn.Module):
    def __init__(self, distance_weight: float = 0.1, variance_weight: float = 0.1):
        super().__init__()
        self.distance_weight = distance_weight
        self.variance_weight = variance_weight

    def forward(self, xt, xt_pred, xp, xp_pred):
        # Reconstruction Loss
        recon = mse_loss(xt_pred, xt) + mse_loss(xp_pred, xp)

        # Distance Loss
        def dist_mtx(x):
            return torch.cdist(x.squeeze(-1), x.squeeze(-1))

        t_loss = mse_loss(dist_mtx(xt_pred), dist_mtx(xt))
        p_loss = mse_loss(dist_mtx(xp_pred), dist_mtx(xp))
        dist =  t_loss + p_loss

        # Variance Loss
        t_var = torch.abs(xt.var() - xt_pred.var())
        p_var = torch.abs(xp.var() - xp_pred.var())
        var = t_var + p_var

        # Composite Loss
        composite = recon + self.distance_weight * dist + self.variance_weight * var
        return composite