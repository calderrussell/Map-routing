from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.baselines import dynamic_shortest_path
from visualization.cli import main as visualization_main
from visualization.folium_results import render_folium_result, trace_summary
from visualization.graph_io import load_graph_problem
from visualization.trace import evaluate_controller_with_trace


ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "examples" / "graphs" / "cambridge_mit.json"


def _baseline_trace():
    problem = load_graph_problem(GRAPH, horizon=5)
    trace = evaluate_controller_with_trace(
        "dynamic_shortest_path",
        problem.simulator,
        problem.scenario,
        lambda state, forecast: dynamic_shortest_path(
            problem.network, state, problem.destinations
        ),
    )
    return problem, trace


def test_geographic_graph_loads_into_reachable_ctm() -> None:
    problem = load_graph_problem(GRAPH, horizon=5)
    assert problem.network.n_cells == 20
    assert problem.network.n_movements == 32
    assert problem.origins.shape == problem.destinations.shape == (2,)
    assert problem.scenario.realized_demand.shape == (5, 2)
    assert len(problem.scenario.incidents) == 1
    assert all(problem.network.reachability_mask(problem.destinations).any(axis=1))


def test_traced_evaluation_retains_physical_model_outputs() -> None:
    problem, trace = _baseline_trace()
    assert len(trace.steps) == 5
    assert trace.steps[0].controlled_action.shape == (
        2,
        problem.network.n_movements,
    )
    assert trace.steps[0].result.movement_flow.shape == (
        2,
        problem.network.n_movements,
    )
    assert trace.evaluation.metrics.capacity_violation == 0.0
    assert trace.evaluation.metrics.invalid_turn_rate == 0.0
    assert trace.evaluation.metrics.conservation_residual < 1e-9


def test_folium_renderer_embeds_results_and_attribution(tmp_path: Path) -> None:
    problem, trace = _baseline_trace()
    destination = render_folium_result(
        problem, trace, tmp_path / "result.html", baseline_trace=trace
    )
    document = destination.read_text()
    assert "gnn-routing-summary" in document
    assert "leaflet.timedimension" in document
    assert "GNN: maximum congestion + routed volume" in document
    assert "OpenStreetMap contributors" in document
    summary = trace_summary(problem, trace, baseline_trace=trace)
    assert summary["cells"] == 20
    assert np.isclose(summary["gnn_minus_baseline_tstt"], 0.0)


def test_cli_trains_evaluates_and_writes_map_summary_and_checkpoint(
    tmp_path: Path,
) -> None:
    output = tmp_path / "model_result.html"
    checkpoint = tmp_path / "model.pt"
    assert (
        visualization_main(
            [
                "--graph",
                str(GRAPH),
                "--train-demo",
                "--output",
                str(output),
                "--save-checkpoint",
                str(checkpoint),
                "--horizon",
                "4",
                "--train-demonstrations",
                "1",
                "--oracle-horizon",
                "2",
                "--oracle-iterations",
                "1",
                "--epochs",
                "1",
                "--hidden-dim",
                "16",
                "--layers",
                "1",
            ]
        )
        == 0
    )
    summary = json.loads(output.with_suffix(".json").read_text())
    assert output.exists() and checkpoint.exists()
    assert summary["controller"] == "heterogeneous_gnn"
    assert summary["model"]["source"] == "trained_for_visual_demo"
    assert summary["folium_html"] == str(output.resolve())
