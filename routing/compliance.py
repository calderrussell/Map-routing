from __future__ import annotations

import numpy as np


def combine_controlled_actions(
    controlled: np.ndarray,
    uncontrolled: np.ndarray,
    controlled_fraction: float,
) -> np.ndarray:
    if controlled.shape != uncontrolled.shape:
        raise ValueError("controlled and uncontrolled actions must share a shape")
    if not 0.0 <= controlled_fraction <= 1.0:
        raise ValueError("controlled fraction must be within [0, 1]")
    action = controlled_fraction * controlled + (1.0 - controlled_fraction) * uncontrolled
    return np.maximum(action, 0.0)


def compliance_probability(
    detour_ratio: np.ndarray,
    trust: np.ndarray,
    *,
    detour_sensitivity: float = 4.0,
    trust_weight: float = 2.0,
) -> np.ndarray:
    """Optional learned-behavior-compatible logistic compliance interface."""

    logits = trust_weight * np.asarray(trust) - detour_sensitivity * (
        np.asarray(detour_ratio) - 1.0
    )
    return 1.0 / (1.0 + np.exp(-logits))

