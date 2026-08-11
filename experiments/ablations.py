from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Ablation:
    name: str
    heterogeneous: bool
    projection: bool
    decision_focused: bool
    od_tokens: bool


ABLATIONS = (
    Ablation("mlp_no_graph", False, False, False, False),
    Ablation("homogeneous_gcn_gru", False, True, True, False),
    Ablation("heterograph_no_projection", True, False, True, True),
    Ablation("heterograph_no_decision_loss", True, True, False, True),
    Ablation("heterograph_no_od_tokens", True, True, True, False),
    Ablation("full", True, True, True, True),
)


def ablation_by_name(name: str) -> Ablation:
    try:
        return next(item for item in ABLATIONS if item.name == name)
    except StopIteration as exc:
        raise KeyError(name) from exc

