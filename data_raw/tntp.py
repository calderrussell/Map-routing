from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from oracle.static_assignment import StaticNetwork


def _data_lines(path: str | Path):
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("~", "<")):
            continue
        yield stripped.rstrip(";")


def load_tntp_network(path: str | Path, name: str | None = None) -> StaticNetwork:
    """Load the standard TNTP link format without modifying the raw source."""

    edges: list[tuple[int, int]] = []
    capacity: list[float] = []
    free_time: list[float] = []
    coefficient: list[float] = []
    power: list[float] = []
    nodes: set[int] = set()
    for line in _data_lines(path):
        columns = re.split(r"\s+", line)
        if len(columns) < 6:
            continue
        source, target = int(columns[0]), int(columns[1])
        cap, fft, alpha, beta = map(float, columns[2:6])
        edges.append((source, target))
        nodes.update((source, target))
        capacity.append(cap)
        free_time.append(fft)
        coefficient.append(fft * alpha)
        power.append(beta)
    if not edges:
        raise ValueError(f"no TNTP links found in {path}")
    return StaticNetwork(
        name=name or Path(path).stem,
        nodes=tuple(sorted(nodes)),
        edges=tuple(edges),
        free_time=np.asarray(free_time),
        capacity=np.asarray(capacity),
        coefficient=np.asarray(coefficient),
        power=np.asarray(power),
    )


def load_tntp_demand(path: str | Path) -> dict[tuple[int, int], float]:
    demand: dict[tuple[int, int], float] = {}
    origin: int | None = None
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"\s*Origin\s+(\d+)", line, flags=re.IGNORECASE)
        if match:
            origin = int(match.group(1))
            continue
        if origin is None or line.lstrip().startswith(("<", "~")):
            continue
        for destination, volume in re.findall(r"(\d+)\s*:\s*([0-9.eE+-]+)", line):
            value = float(volume)
            if value > 0:
                demand[(origin, int(destination))] = value
    if not demand:
        raise ValueError(f"no TNTP demand records found in {path}")
    return demand

