from __future__ import annotations

import torch


def scatter_mean(messages: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    output = torch.zeros((size, messages.shape[-1]), device=messages.device, dtype=messages.dtype)
    output.index_add_(0, index, messages)
    count = torch.zeros(size, device=messages.device, dtype=messages.dtype)
    count.index_add_(0, index, torch.ones_like(index, dtype=messages.dtype))
    return output / count.clamp_min(1.0).unsqueeze(-1)


def movement_softmax(
    logits: torch.Tensor,
    movement_sources: torch.Tensor,
    legal_mask: torch.Tensor,
    n_cells: int,
) -> torch.Tensor:
    """Masked softmax over admissible outgoing movements for each commodity/cell."""

    if logits.shape != legal_mask.shape:
        raise ValueError("logits and legal_mask must share K x M shape")
    output = torch.zeros_like(logits)
    masked = logits.masked_fill(~legal_mask, -1e9)
    for cell in range(n_cells):
        indices = torch.nonzero(movement_sources == cell, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        local_legal = legal_mask[:, indices]
        local = torch.softmax(masked[:, indices], dim=-1) * local_legal
        local = local / local.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        output[:, indices] = local
    return output

