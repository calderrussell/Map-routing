from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.common import movement_softmax, scatter_mean
from models.features import FeatureBatch, CELL_FEATURES, MOVEMENT_FEATURES, OD_FEATURES


@dataclass
class ModelOutput:
    logits: torch.Tensor
    splits: torch.Tensor
    hidden: torch.Tensor


class GraphResidualLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.forward_message = nn.Linear(hidden_dim, hidden_dim)
        self.reverse_message = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        incoming = scatter_mean(self.forward_message(hidden[source]), target, hidden.shape[0])
        outgoing = scatter_mean(self.reverse_message(hidden[target]), source, hidden.shape[0])
        return self.norm(hidden + self.update(torch.cat([hidden, incoming, outgoing], dim=-1)))


class DestinationGCNGRU(nn.Module):
    """Phase 1 destination-conditioned same-topology GCN/GRU policy."""

    def __init__(self, hidden_dim: int = 64, layers: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell_encoder = nn.Sequential(
            nn.Linear(len(CELL_FEATURES), hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.spatial = nn.ModuleList(GraphResidualLayer(hidden_dim) for _ in range(layers))
        self.temporal = nn.GRUCell(hidden_dim, hidden_dim)
        self.od_encoder = nn.Sequential(
            nn.Linear(len(OD_FEATURES), hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.movement_encoder = nn.Sequential(
            nn.Linear(len(MOVEMENT_FEATURES), hidden_dim // 2), nn.SiLU()
        )
        decoder_dim = hidden_dim * 4 + hidden_dim // 2
        self.decoder = nn.Sequential(
            nn.Linear(decoder_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, batch: FeatureBatch, previous_hidden: torch.Tensor | None = None
    ) -> ModelOutput:
        hidden = self.cell_encoder(batch.cell)
        for layer in self.spatial:
            hidden = layer(hidden, batch.edge_index)
        if previous_hidden is None:
            previous_hidden = torch.zeros_like(hidden)
        hidden = self.temporal(hidden, previous_hidden)
        od = self.od_encoder(batch.od) + hidden[batch.destinations]
        movement = self.movement_encoder(batch.movement)
        k = batch.od.shape[0]
        m = batch.movement.shape[0]
        source_hidden = hidden[batch.movement_sources].unsqueeze(0).expand(k, m, -1)
        target_hidden = hidden[batch.movement_targets].unsqueeze(0).expand(k, m, -1)
        origin_hidden = hidden[batch.origins].unsqueeze(1).expand(k, m, -1)
        od_hidden = od.unsqueeze(1).expand(k, m, -1)
        movement_hidden = movement.unsqueeze(0).expand(k, m, -1)
        logits = self.decoder(
            torch.cat(
                [source_hidden, target_hidden, origin_hidden, od_hidden, movement_hidden], dim=-1
            )
        ).squeeze(-1)
        splits = movement_softmax(
            logits,
            batch.movement_sources,
            batch.legal_mask,
            batch.cell.shape[0],
        )
        return ModelOutput(logits=logits, splits=splits, hidden=hidden)

