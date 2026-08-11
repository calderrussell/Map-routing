# Phase 4 audit — robustness and independent simulation surface

Specification source: `specification/gnn_dso_model.tex`, lines 774–925 and 966–971.

| Requirement | Implementation/evidence | Status |
|---|---|---|
| K-path loop-free route generation | `routing/paths.py` | Implemented/tested |
| Path-flow decomposition with switching penalty | SLSQP simplex program matching equation (24) | Exact on smoke target |
| Balanced driver assignment | Largest-remainder rounding with seeded ties | 101/101 assigned |
| Next-hop method and loop prevention | Visited-set rejection plus route commitment manager | Implemented/tested |
| Partial control | Controlled/uncontrolled action mixture and logistic compliance interface | Implemented |
| All local baselines | Free/dynamic shortest path, static UE/SO/marginal cost, DSO, short MPC, backpressure, MLP, homogeneous and ablations | Implemented/registered |
| External graph baseline | Explicit reproducibility/availability adapter | Implemented |
| E0–E7 | `configs/frozen_experiment_plan.yaml` | Complete and validated |
| Robustness axes | Demand, capacity, forecast, sensor noise, delay, controlled fraction | 22 conditions |
| Metrics | System, optimality, feasibility, robustness, full latency, fairness | Implemented |
| Statistical design | Paired common seeds, bootstrap CI/effect size, 30 seeds | Implemented/frozen |
| SUMO transfer interface | `simulators/sumo_adapter.py`, fake-TraCI test | Interface passes |
| Independent SUMO run | Requires external SUMO binary/network | Not executed: binary absent |

The smoke incident/penetration sweep produced finite outputs at controlled fractions
0, 0.25, 0.5, 0.75, and 1.0, with zero CTM capacity violation and maximum conservation
residual 7.11e-15. These smoke costs are diagnostics, not paper results.

SUMO status is recorded as `SUMO binary 'sumo' is not installed`. The project refuses
to substitute CTM evidence for E5 microscopic transfer evidence.

