from __future__ import annotations

import torch
from torch import nn

from models.features import CELL_FEATURES, MOVEMENT_FEATURES, OD_FEATURES, FeatureBatch


class LearnedMarginalCostDecoder(nn.Module):
    """Optional nonnegative dynamic externality decoder (TeX equation 15)."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(len(CELL_FEATURES) * 2 + len(OD_FEATURES) + len(MOVEMENT_FEATURES), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: FeatureBatch) -> tuple[torch.Tensor, torch.Tensor]:
        k, m = batch.od.shape[0], batch.movement.shape[0]
        features = torch.cat(
            [
                batch.cell[batch.movement_sources].unsqueeze(0).expand(k, m, -1),
                batch.cell[batch.movement_targets].unsqueeze(0).expand(k, m, -1),
                batch.od.unsqueeze(1).expand(k, m, -1),
                batch.movement.unsqueeze(0).expand(k, m, -1),
            ],
            dim=-1,
        )
        externality = torch.nn.functional.softplus(self.decoder(features).squeeze(-1))
        estimated_time = batch.cell[batch.movement_targets, 4].unsqueeze(0)
        return externality, estimated_time + externality


class FullMulticommodityFlowDecoder(nn.Module):
    """Optional offline full-flow proposal; it must be followed by projection."""

    def __init__(self, hidden_dim: int = 64, horizon: int = 4) -> None:
        super().__init__()
        self.horizon = horizon
        self.decoder = nn.Sequential(
            nn.Linear(len(CELL_FEATURES) * 2 + len(OD_FEATURES) + len(MOVEMENT_FEATURES), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, batch: FeatureBatch) -> torch.Tensor:
        k, m = batch.od.shape[0], batch.movement.shape[0]
        features = torch.cat(
            [
                batch.cell[batch.movement_sources].unsqueeze(0).expand(k, m, -1),
                batch.cell[batch.movement_targets].unsqueeze(0).expand(k, m, -1),
                batch.od.unsqueeze(1).expand(k, m, -1),
                batch.movement.unsqueeze(0).expand(k, m, -1),
            ],
            dim=-1,
        )
        flow = torch.nn.functional.softplus(self.decoder(features))
        return flow.permute(2, 0, 1) * batch.legal_mask.unsqueeze(0)
