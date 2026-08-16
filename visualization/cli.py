from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from data_processed.dataset import gradient_oracle_demonstrations
from experiments.baselines import dynamic_shortest_path
from experiments.evaluate import incident_arrays
from experiments.train import set_deterministic_seed, train_imitation
from models.features import build_features
from models.heterogeneous import HeteroSpatioTemporalGNN
from visualization.folium_results import render_folium_result, trace_summary
from visualization.graph_io import GeoGraphProblem, load_graph_problem
from visualization.trace import evaluate_controller_with_trace


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH = REPOSITORY_ROOT / "examples" / "graphs" / "cambridge_mit.json"


class GeographicGNNPolicy:
    """Stateful GNN policy whose visible incidents match the traced simulator state."""

    def __init__(self, model: HeteroSpatioTemporalGNN, problem: GeoGraphProblem) -> None:
        self.model = model
        self.problem = problem
        self.hidden = None

    def __call__(self, state, forecast):
        cap, speed, disabled = incident_arrays(
            self.problem.simulator, self.problem.scenario, state.time
        )
        features = build_features(
            self.problem.network,
            state,
            self.problem.origins,
            self.problem.destinations,
            forecast,
            capacity_multiplier=cap,
            speed_multiplier=speed,
            disabled_movements=disabled,
            forecast_confidence=0.85,
        )
        self.model.eval()
        with torch.no_grad():
            output = self.model(features, self.hidden)
        self.hidden = output.hetero_hidden
        return output.splits.cpu().numpy()


def _infer_model_config(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    try:
        hidden_dim = int(state_dict["cell_encoder.0.weight"].shape[0])
    except (KeyError, AttributeError, IndexError) as exc:
        raise ValueError("checkpoint is not a HeteroSpatioTemporalGNN state dictionary") from exc
    relation_indices = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("relations.") and key.split(".")[1].isdigit()
    }
    if not relation_indices:
        raise ValueError("checkpoint has no heterogeneous relation layers")
    return {"hidden_dim": hidden_dim, "layers": max(relation_indices) + 1}


def _load_model(path: Path) -> tuple[HeteroSpatioTemporalGNN, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint has no state_dict")
    inferred = _infer_model_config(state_dict)
    supplied = payload.get("model_config", {})
    config = {**inferred, **supplied} if isinstance(supplied, dict) else inferred
    model = HeteroSpatioTemporalGNN(
        hidden_dim=int(config["hidden_dim"]), layers=int(config["layers"])
    )
    model.load_state_dict(state_dict)
    return model, {
        "source": "checkpoint",
        "checkpoint": str(path.resolve()),
        "model_config": config,
    }


def _train_demo_model(
    problem: GeoGraphProblem,
    *,
    seed: int,
    demonstrations: int,
    oracle_horizon: int,
    oracle_iterations: int,
    epochs: int,
    hidden_dim: int,
    layers: int,
) -> tuple[HeteroSpatioTemporalGNN, dict[str, object]]:
    set_deterministic_seed(seed)
    started = time.perf_counter()
    training_data = gradient_oracle_demonstrations(
        problem.network,
        problem.origins,
        problem.destinations,
        count=demonstrations,
        horizon=oracle_horizon,
        seed=seed,
        oracle_iterations=oracle_iterations,
    )
    model = HeteroSpatioTemporalGNN(hidden_dim=hidden_dim, layers=layers)
    history = train_imitation(
        model,
        training_data,
        epochs=epochs,
        learning_rate=2e-3,
        seed=seed,
    )
    return model, {
        "source": "trained_for_visual_demo",
        "model_config": {"hidden_dim": hidden_dim, "layers": layers},
        "demonstrations": demonstrations,
        "oracle_horizon": oracle_horizon,
        "oracle_iterations": oracle_iterations,
        "oracle_statuses": sorted({item.diagnostics.status for item in training_data}),
        "oracle_certified": all(
            item.diagnostics.certified_gap == 0.0 for item in training_data
        ),
        "epochs": epochs,
        "loss_initial": history.losses[0],
        "loss_final": history.losses[-1],
        "training_seconds": time.perf_counter() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the routing GNN on a geographic graph and write a Folium result map."
    )
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Previously trained GNN checkpoint")
    source.add_argument(
        "--train-demo",
        action="store_true",
        help="Train a small local oracle-imitation model before evaluating",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/runs/folium_gnn_result.html")
    )
    parser.add_argument("--summary", type=Path, help="JSON path; defaults beside the HTML")
    parser.add_argument("--save-checkpoint", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--controlled-fraction", type=float, default=1.0)
    parser.add_argument("--baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-demonstrations", type=int, default=3)
    parser.add_argument("--oracle-horizon", type=int, default=6)
    parser.add_argument("--oracle-iterations", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--layers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.controlled_fraction <= 1.0:
        raise SystemExit("--controlled-fraction must lie in [0, 1]")
    for label in ("train_demonstrations", "oracle_horizon", "oracle_iterations", "epochs"):
        if getattr(args, label) < 1:
            raise SystemExit(f"--{label.replace('_', '-')} must be positive")
    problem = load_graph_problem(args.graph, seed=args.seed, horizon=args.horizon)
    seed = problem.scenario.seed if args.seed is None else args.seed
    if args.checkpoint is not None:
        model, provenance = _load_model(args.checkpoint)
    else:
        model, provenance = _train_demo_model(
            problem,
            seed=seed,
            demonstrations=args.train_demonstrations,
            oracle_horizon=args.oracle_horizon,
            oracle_iterations=args.oracle_iterations,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
        )
    model_trace = evaluate_controller_with_trace(
        "heterogeneous_gnn",
        problem.simulator,
        problem.scenario,
        GeographicGNNPolicy(model, problem),
        controlled_fraction=args.controlled_fraction,
    )
    baseline_trace = None
    if args.baseline:
        baseline_trace = evaluate_controller_with_trace(
            "dynamic_shortest_path",
            problem.simulator,
            problem.scenario,
            lambda state, forecast: dynamic_shortest_path(
                problem.network, state, problem.destinations
            ),
        )
    output = render_folium_result(
        problem,
        model_trace,
        args.output,
        baseline_trace=baseline_trace,
    )
    summary = trace_summary(problem, model_trace, baseline_trace=baseline_trace)
    summary["model"] = provenance
    summary["folium_html"] = str(output)
    summary_path = (
        args.summary.expanduser().resolve()
        if args.summary is not None
        else output.with_suffix(".json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.save_checkpoint is not None:
        checkpoint = args.save_checkpoint.expanduser().resolve()
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_config": provenance["model_config"],
                "graph": problem.graph.name,
                "seed": seed,
                "visual_evaluation": summary,
            },
            checkpoint,
        )
        summary["saved_checkpoint"] = str(checkpoint)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
