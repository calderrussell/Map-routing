from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from data_processed.scenarios import assert_topology_disjoint


SplitName = Literal["development", "train", "validation", "test"]


@dataclass(frozen=True)
class TopologySplits:
    development: tuple[str, ...]
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "TopologySplits":
        payload = yaml.safe_load(Path(path).read_text())
        required = {"development", "train", "validation", "test"}
        if set(payload) != required:
            raise ValueError(f"split file must contain exactly {sorted(required)}")
        assert_topology_disjoint(payload)
        return cls(**{key: tuple(payload[key]) for key in required})

    def split_of(self, topology: str) -> SplitName:
        for split in ("development", "train", "validation", "test"):
            if topology in getattr(self, split):
                return split  # type: ignore[return-value]
        raise KeyError(f"topology {topology!r} is not assigned to a split")

    def authorize(self, topology: str, purpose: Literal["fit", "tune", "final_test"]) -> None:
        split = self.split_of(topology)
        allowed = {
            "fit": {"development", "train"},
            "tune": {"development", "validation"},
            "final_test": {"test"},
        }[purpose]
        if split not in allowed:
            raise PermissionError(
                f"topology {topology!r} is in {split}, not authorized for {purpose}"
            )

