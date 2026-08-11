from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn

from data_processed.dataset import Demonstration
from models.homogeneous import DestinationGCNGRU


@dataclass(frozen=True)
class TrainingHistory:
    losses: tuple[float, ...]
    best_loss: float
    epochs: int


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def imitation_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Normalized Huber loss from TeX equation (20), restricted to legal turns."""

    values = nn.functional.huber_loss(predicted, target, reduction="none", delta=0.1)
    weights = legal_mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def train_imitation(
    model: DestinationGCNGRU,
    demonstrations: list[Demonstration],
    *,
    epochs: int = 120,
    learning_rate: float = 3e-3,
    seed: int = 0,
) -> TrainingHistory:
    if not demonstrations:
        raise ValueError("at least one demonstration is required")
    set_deterministic_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    losses = []
    best = float("inf")
    for _ in range(epochs):
        order = np.random.permutation(len(demonstrations))
        total = 0.0
        for index in order:
            item = demonstrations[int(index)]
            optimizer.zero_grad()
            output = model(item.features)
            target = torch.as_tensor(item.target_action, dtype=output.splits.dtype)
            loss = imitation_loss(output.splits, target, item.features.legal_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
        epoch_loss = total / len(demonstrations)
        losses.append(epoch_loss)
        best = min(best, epoch_loss)
    return TrainingHistory(tuple(losses), best, epochs)

