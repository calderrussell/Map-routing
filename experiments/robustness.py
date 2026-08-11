from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobustnessCondition:
    name: str
    demand_scale: float = 1.0
    capacity_loss: float = 0.0
    forecast_error: float = 0.0
    sensor_noise: float = 0.0
    control_delay: int = 0
    controlled_fraction: float = 1.0


def robustness_grid() -> tuple[RobustnessCondition, ...]:
    conditions = [RobustnessCondition("nominal")]
    conditions.extend(
        RobustnessCondition(f"demand_{scale:.2f}", demand_scale=scale)
        for scale in (0.75, 1.25, 1.5)
    )
    conditions.extend(
        RobustnessCondition(f"capacity_loss_{loss:.2f}", capacity_loss=loss)
        for loss in (0.1, 0.3, 0.6, 1.0)
    )
    conditions.extend(
        RobustnessCondition(f"forecast_error_{error:.2f}", forecast_error=error)
        for error in (0.1, 0.25, 0.5)
    )
    conditions.extend(
        RobustnessCondition(f"sensor_noise_{noise:.2f}", sensor_noise=noise)
        for noise in (0.05, 0.15, 0.3)
    )
    conditions.extend(
        RobustnessCondition(f"control_delay_{delay}", control_delay=delay)
        for delay in (1, 2, 4)
    )
    conditions.extend(
        RobustnessCondition(f"controlled_{fraction:.2f}", controlled_fraction=fraction)
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    return tuple(conditions)

