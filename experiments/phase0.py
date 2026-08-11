from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from data_processed.networks import analytic_parallel_static, braess_static, sioux_falls_static
from oracle.static_assignment import frank_wolfe, price_of_anarchy_gap


@dataclass(frozen=True)
class Phase0Result:
    analytic_parallel_ue_tstt: float
    analytic_parallel_so_tstt: float
    braess_ue_tstt: float
    braess_so_tstt: float
    braess_removed_ue_tstt: float
    braess_ue_gap: float
    braess_so_gap: float
    sioux_falls_nodes: int
    sioux_falls_links: int
    sioux_falls_ue_tstt: float
    sioux_falls_so_tstt: float


def run_phase0() -> Phase0Result:
    parallel, parallel_demand = analytic_parallel_static()
    parallel_ue = frank_wolfe(parallel, parallel_demand, "ue")
    parallel_so = frank_wolfe(parallel, parallel_demand, "so")
    braess, demand = braess_static(True)
    removed, _ = braess_static(False)
    braess_ue = frank_wolfe(braess, demand, "ue")
    braess_so = frank_wolfe(braess, demand, "so")
    removed_ue = frank_wolfe(removed, demand, "ue")
    sioux, sioux_demand = sioux_falls_static()
    sioux_ue = frank_wolfe(sioux, sioux_demand, "ue", max_iterations=300, tolerance=1e-6)
    sioux_so = frank_wolfe(sioux, sioux_demand, "so", max_iterations=300, tolerance=1e-6)
    return Phase0Result(
        parallel_ue.tstt,
        parallel_so.tstt,
        braess_ue.tstt,
        braess_so.tstt,
        removed_ue.tstt,
        price_of_anarchy_gap(braess_ue.tstt, braess_ue.tstt, braess_so.tstt),
        price_of_anarchy_gap(braess_so.tstt, braess_ue.tstt, braess_so.tstt),
        len(sioux.nodes),
        sioux.n_edges,
        sioux_ue.tstt,
        sioux_so.tstt,
    )


def write_phase0_result(result: Phase0Result, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")

