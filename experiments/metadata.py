from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import cvxpy
import networkx
import numpy
import scipy
import torch


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReproducibilityMetadata:
    source_revision: str
    source_dirty: bool
    configuration_sha256: str
    network_sha256: str
    random_seeds: tuple[int, ...]
    solver_versions: dict[str, str]
    solver_tolerances: dict[str, float]
    hardware: str
    checkpoint_sha256: str | None


def collect_metadata(
    config_path: str | Path,
    network_hash: str,
    seeds: tuple[int, ...],
    *,
    checkpoint_path: str | Path | None = None,
    solver_tolerances: dict[str, float] | None = None,
) -> ReproducibilityMetadata:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.CalledProcessError:
        revision = "unborn"
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout)
    return ReproducibilityMetadata(
        source_revision=revision,
        source_dirty=dirty,
        configuration_sha256=sha256_file(config_path) or "",
        network_sha256=network_hash,
        random_seeds=seeds,
        solver_versions={
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "networkx": networkx.__version__,
            "torch": torch.__version__,
            "cvxpy": cvxpy.__version__,
        },
        solver_tolerances=solver_tolerances or {"eps_abs": 1e-8, "eps_rel": 1e-8},
        hardware=f"{platform.system()} {platform.machine()} | {platform.processor()}",
        checkpoint_sha256=sha256_file(checkpoint_path),
    )


def write_metadata(metadata: ReproducibilityMetadata, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n")

