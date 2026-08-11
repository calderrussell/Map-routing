# Graph Neural Surrogate for Dynamic System-Optimal Traffic Assignment

This repository implements the complete research/software blueprint in
[`specification/gnn_dso_model.tex`](specification/gnn_dso_model.tex). It is a routing
decision system: a heterogeneous spatio-temporal GNN proposes commodity-specific
movement decisions, hard projection enforces traffic physics, and a CTM oracle supplies
receding-horizon system-optimal demonstrations.

## Current verified status

- Phase 0: analytic UE/SO, Braess, 24-node/76-link Sioux Falls, finite-storage
  multicommodity CTM, source queues, certified tiny oracle, and uncertified local DSO.
- Phase 1: same-topology GCN/GRU smoke test improves TSTT 37.5% over dynamic shortest
  paths and is 6.31% above the certified discrete oracle on its fixed held-out profile.
- Phase 2: the cell/intersection/OD/global heterograph trains across two graphs and runs
  with zero illegal flow on a topology with a hash excluded from training.
- Phase 3: OSQP projection eliminates hard violations; decision-focused fine-tuning
  sharply reduces raw violation and projection reliance in the smoke fixture.
- Phase 4: E0–E7, 30 paired seeds, robustness/penetration sweeps, path decomposition,
  fairness/statistics, reproducibility metadata, and a TraCI bridge are implemented.
  A real E5 SUMO run is not claimed because SUMO and a calibrated network are absent.

These are correctness/integration results, not the final paper study. The project will
only support broad near-optimality, topology-generalization, robustness, scalability,
or microscopic-transfer claims after the frozen full experiment plan is executed.

## Setup

Python 3.11+ is required.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

For SUMO transfer, install SUMO separately and then install the optional TraCI extra:

```bash
.venv/bin/pip install -e '.[sumo]'
```

## Run

```bash
.venv/bin/python -m experiments.run phase0
.venv/bin/python -m experiments.run phase1
.venv/bin/python -m experiments.run phase2
.venv/bin/python -m experiments.run phase3
.venv/bin/python -m experiments.run phase4
.venv/bin/python -m experiments.run all-smoke
.venv/bin/python -m experiments.run test
```

The frozen protocol is [`configs/frozen_experiment_plan.yaml`](configs/frozen_experiment_plan.yaml).
It assigns entire topologies before scenario generation, includes all E0–E7 conditions,
lists every baseline and ablation, and predeclares primary metrics/statistics. The code
rejects using final-test topologies for fitting or tuning.

## Project map

```text
configs/             frozen experiment and network configurations
data_raw/            immutable TNTP/OSM/SUMO inputs and parsers
data_processed/      cell graphs, scenarios, splits, demonstrations
oracle/              static assignment, CTM DSO, exhaustive certification
models/              GCN/GRU, heterogeneous GNN, MLP and alternative decoders
projection/          exact QP and differentiable feasibility layers
routing/             K paths, path decomposition, rounding, compliance
simulators/          NumPy/PyTorch CTM and SUMO/TraCI adapter
experiments/         training, DAgger, evaluation, metrics, phases, CLI
tests/               correctness, leakage, projection, model and integration tests
artifacts/           milestone and final compliance audits
specification/       recovered authoritative LaTeX design and bibliography
```

See [`docs/implementation_spec.md`](docs/implementation_spec.md) for the equation-to-code
map and the milestone audit files in [`artifacts/`](artifacts/README.md) for evidence and
claim boundaries.

