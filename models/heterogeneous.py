from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.common import movement_softmax, scatter_mean
from models.features import (
    CELL_FEATURES,
    GLOBAL_FEATURES,
    INTERSECTION_FEATURES,
    MOVEMENT_FEATURES,
    OD_FEATURES,
    FeatureBatch,
)
from models.homogeneous import ModelOutput


@dataclass
class HeteroHidden:
    cell: torch.Tensor
    intersection: torch.Tensor
    od: torch.Tensor


class RelationLayer(nn.Module):
    """Directional cell, virtual OD, and global relation-specific messages."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.forward_message = nn.Linear(hidden_dim * 2, hidden_dim)
        self.reverse_message = nn.Linear(hidden_dim * 2, hidden_dim)
        self.od_to_origin = nn.Linear(hidden_dim, hidden_dim)
        self.cell_to_intersection = nn.Linear(hidden_dim, hidden_dim)
        self.intersection_to_cell = nn.Linear(hidden_dim, hidden_dim)
        self.destination_to_od = nn.Linear(hidden_dim, hidden_dim)
        self.origin_to_od = nn.Linear(hidden_dim, hidden_dim)
        self.global_to_cell = nn.Linear(hidden_dim, hidden_dim)
        self.global_to_od = nn.Linear(hidden_dim, hidden_dim)
        self.cell_gate = nn.Linear(hidden_dim * 2, 1)
        self.od_gate = nn.Linear(hidden_dim * 2, 1)
        self.cell_update = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.intersection_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.od_update = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.cell_norm = nn.LayerNorm(hidden_dim)
        self.od_norm = nn.LayerNorm(hidden_dim)
        self.intersection_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        cell: torch.Tensor,
        intersection: torch.Tensor,
        od: torch.Tensor,
        global_hidden: torch.Tensor,
        batch: FeatureBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source, target = batch.edge_index
        forward_pair = torch.cat([cell[source], cell[target]], dim=-1)
        reverse_pair = torch.cat([cell[target], cell[source]], dim=-1)
        forward_gate = torch.sigmoid(self.cell_gate(forward_pair))
        reverse_gate = torch.sigmoid(self.cell_gate(reverse_pair))
        incoming = scatter_mean(
            self.forward_message(forward_pair) * forward_gate, target, cell.shape[0]
        )
        outgoing = scatter_mean(
            self.reverse_message(reverse_pair) * reverse_gate, source, cell.shape[0]
        )
        od_messages = self.od_to_origin(od)
        od_weight = torch.sigmoid(self.od_gate(torch.cat([od, cell[batch.origins]], dim=-1)))
        origin_injection = torch.zeros_like(cell).index_add(
            0, batch.origins, od_messages * od_weight
        )
        global_cell = self.global_to_cell(global_hidden).expand_as(cell)
        intersection_message = self.intersection_to_cell(intersection)
        cell_delta = self.cell_update(
            torch.cat(
                [cell, incoming, outgoing, origin_injection, intersection_message, global_cell],
                dim=-1,
            )
        )
        next_cell = self.cell_norm(cell + cell_delta)
        next_intersection = self.intersection_norm(
            intersection
            + self.intersection_update(
                torch.cat([intersection, self.cell_to_intersection(next_cell)], dim=-1)
            )
        )
        destination_context = self.destination_to_od(next_cell[batch.destinations])
        origin_context = self.origin_to_od(next_cell[batch.origins])
        global_od = self.global_to_od(global_hidden).expand_as(od)
        od_delta = self.od_update(
            torch.cat([od, origin_context, destination_context, global_od], dim=-1)
        )
        next_od = self.od_norm(od + od_delta)
        return next_cell, next_intersection, next_od


class HeteroSpatioTemporalGNN(nn.Module):
    """OD-aware heterogeneous topology-general routing policy (Phase 2)."""

    def __init__(self, hidden_dim: int = 64, layers: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell_encoder = nn.Sequential(
            nn.Linear(len(CELL_FEATURES), hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.od_encoder = nn.Sequential(
            nn.Linear(len(OD_FEATURES), hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.intersection_encoder = nn.Sequential(
            nn.Linear(len(INTERSECTION_FEATURES), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(len(GLOBAL_FEATURES), hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.relations = nn.ModuleList(RelationLayer(hidden_dim) for _ in range(layers))
        self.cell_temporal = nn.GRUCell(hidden_dim, hidden_dim)
        self.intersection_temporal = nn.GRUCell(hidden_dim, hidden_dim)
        self.od_temporal = nn.GRUCell(hidden_dim, hidden_dim)
        self.movement_encoder = nn.Sequential(
            nn.Linear(len(MOVEMENT_FEATURES), hidden_dim // 2), nn.SiLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 4 + hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, batch: FeatureBatch, previous_hidden: HeteroHidden | None = None
    ) -> ModelOutput:
        cell = self.cell_encoder(batch.cell)
        intersection = self.intersection_encoder(batch.intersection)
        od = self.od_encoder(batch.od)
        global_hidden = self.global_encoder(batch.global_context).unsqueeze(0)
        for relation in self.relations:
            cell, intersection, od = relation(cell, intersection, od, global_hidden, batch)
        if previous_hidden is None:
            previous_hidden = HeteroHidden(
                torch.zeros_like(cell), torch.zeros_like(intersection), torch.zeros_like(od)
            )
        cell = self.cell_temporal(cell, previous_hidden.cell)
        intersection = self.intersection_temporal(
            intersection, previous_hidden.intersection
        )
        od = self.od_temporal(od, previous_hidden.od)
        movement = self.movement_encoder(batch.movement)
        k, m = od.shape[0], batch.movement.shape[0]
        source_hidden = cell[batch.movement_sources].unsqueeze(0).expand(k, m, -1)
        target_hidden = cell[batch.movement_targets].unsqueeze(0).expand(k, m, -1)
        origin_hidden = cell[batch.origins].unsqueeze(1).expand(k, m, -1)
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
        # ModelOutput.hidden remains a Tensor for the Phase-1 interface.  The richer
        # recurrent state is exposed separately for callers that roll heterogeneous time series.
        output = ModelOutput(logits=logits, splits=splits, hidden=cell)
        output.hetero_hidden = HeteroHidden(cell, intersection, od)  # type: ignore[attr-defined]
        return output


def active_od_indices(demand_forecast: torch.Tensor) -> torch.Tensor:
    """OD token sparsification: retain commodities active anywhere in the horizon."""

    return torch.nonzero(demand_forecast.sum(dim=0) > 0, as_tuple=False).flatten()


def destination_batches(destinations: torch.Tensor, maximum_unique: int) -> list[torch.Tensor]:
    """Commodity-scaling strategy: group OD tokens by destination batches."""

    unique = torch.unique(destinations, sorted=True)
    batches = []
    for start in range(0, len(unique), maximum_unique):
        selected = unique[start : start + maximum_unique]
        batches.append(torch.nonzero(torch.isin(destinations, selected), as_tuple=False).flatten())
    return batches
