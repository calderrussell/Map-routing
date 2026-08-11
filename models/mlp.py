from __future__ import annotations

import torch
from torch import nn

from models.common import movement_softmax
from models.features import CELL_FEATURES, GLOBAL_FEATURES, MOVEMENT_FEATURES, OD_FEATURES, FeatureBatch
from models.homogeneous import ModelOutput


class LocalMLPPolicy(nn.Module):
    """Non-message-passing parameter-matched baseline."""

    def __init__(self, hidden_dim: int = 96) -> None:
        super().__init__()
        input_dim = len(CELL_FEATURES) * 2 + len(OD_FEATURES) + len(MOVEMENT_FEATURES) + len(GLOBAL_FEATURES)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: FeatureBatch, previous_hidden=None) -> ModelOutput:
        k, m = batch.od.shape[0], batch.movement.shape[0]
        source = batch.cell[batch.movement_sources].unsqueeze(0).expand(k, m, -1)
        target = batch.cell[batch.movement_targets].unsqueeze(0).expand(k, m, -1)
        od = batch.od.unsqueeze(1).expand(k, m, -1)
        movement = batch.movement.unsqueeze(0).expand(k, m, -1)
        global_context = batch.global_context.view(1, 1, -1).expand(k, m, -1)
        logits = self.network(torch.cat([source, target, od, movement, global_context], dim=-1)).squeeze(-1)
        splits = movement_softmax(
            logits, batch.movement_sources, batch.legal_mask, batch.cell.shape[0]
        )
        return ModelOutput(logits, splits, torch.empty(0, device=logits.device))

