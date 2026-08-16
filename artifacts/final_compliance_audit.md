# Final requirement-by-requirement compliance audit

Authoritative specification: `specification/gnn_dso_model.tex`.

Full verification result: **34 passed in 9.57s**, plus successful installed `gnn-dso all-smoke`, Sioux Falls example, and Cambridge Folium command runs.

| ID | Requirement | Implementation | Verification | Status |
|---|---|---|---|---|
| C1 | Receding-horizon CTM DSO oracle | `oracle/dso.py` | `tests/test_oracle.py` | implemented |
| C2 | Heterogeneous cell/intersection/OD/global spatio-temporal GNN | `models/heterogeneous.py` | `tests/test_heterogeneous.py` | implemented |
| C3 | Hard feasibility and differentiable projection | `projection/feasible.py` | `tests/test_projection.py` | implemented |
| C4 | Topology-held-out evaluation protocol | `configs/frozen_experiment_plan.yaml` | `tests/test_metrics_matrix_metadata.py` | implemented |
| P0-STATIC | Static UE/SO, Braess, Sioux Falls, normalized gap | `oracle/static_assignment.py` | `tests/test_static_assignment.py` | implemented |
| P0-CTM | Conservation, exits, source queues, capacity | `simulators/ctm.py` | `tests/test_ctm.py` | implemented |
| OUTPUT-1 | Local movement split logits and reachability mask | `models/homogeneous.py` | `tests/test_features_models.py` | implemented |
| OUTPUT-2 | Learned nonnegative marginal costs | `models/alternatives.py` | `tests/test_features_models.py` | implemented |
| OUTPUT-3 | Full multicommodity flow benchmark decoder | `models/alternatives.py` | `tests/test_features_models.py` | implemented |
| FEATURES | Physical normalized feature schema without test leakage | `models/features.py` | `tests/test_features_models.py` | implemented |
| COMMODITY | Sparse OD tokens and destination batching | `models/heterogeneous.py` | `tests/test_heterogeneous.py` | implemented |
| DATA | Dynamic demand, five incident sets, topology-first split | `data_processed/scenarios.py` | `tests/test_scenarios.py` | implemented |
| DEMO | Warm-up and sequential first-action oracle demonstrations | `data_processed/dataset.py` | `tests/test_alternatives_dagger.py` | implemented |
| TRAIN-1 | Normalized Huber imitation | `experiments/train.py` | `tests/test_features_models.py` | implemented |
| TRAIN-2 | Physics residuals and decision-focused rollout curriculum | `experiments/decision_training.py` | `tests/test_decision_training.py` | implemented |
| DAGGER | On-policy prioritized oracle querying | `experiments/decision_training.py` | `tests/test_alternatives_dagger.py` | implemented |
| ROUTING | Path decomposition, rounding, next-hop loop control | `routing/paths.py` | `tests/test_routing.py` | implemented |
| COMPLIANCE | Partial control and compliance interface | `routing/compliance.py` | `tests/test_routing.py` | implemented |
| BASELINES | Optimization, routing, MLP, GCN and external baseline adapters | `experiments/baselines.py` | `tests/test_phase4.py` | implemented |
| E0-E7 | Frozen complete experiment matrix | `configs/frozen_experiment_plan.yaml` | `tests/test_metrics_matrix_metadata.py` | implemented |
| METRICS | System, optimality, feasibility, latency, fairness | `experiments/metrics.py` | `tests/test_metrics_matrix_metadata.py` | implemented |
| STATISTICS | Paired seeds, bootstrap CI, effect sizes | `experiments/metrics.py` | `tests/test_metrics_matrix_metadata.py` | implemented |
| ROBUSTNESS | Demand/capacity/forecast/noise/delay/control sweeps | `experiments/robustness.py` | `tests/test_phase4.py` | implemented |
| SCALING | Network/commodity runtime and memory curves | `experiments/scaling.py` | `tests/test_heterogeneous.py` | implemented |
| REPRO | Revision/config/network/seeds/solver/hardware/checkpoint metadata | `experiments/metadata.py` | `tests/test_metrics_matrix_metadata.py` | implemented |
| SUMO | Frozen-controller TraCI transfer adapter | `simulators/sumo_adapter.py` | `tests/test_sumo_adapter.py` | implemented; external run blocked by absent SUMO binary/network |
| REAL-EXAMPLE | End-to-end Sioux Falls GNN walkthrough | `examples/sioux_falls_gnn.py` | `tests/test_real_example.py` | implemented |
| VISUALIZATION | Coordinate-file import, traced GNN evaluation, and Folium result map | `visualization/cli.py` | `tests/test_folium_visualization.py` | implemented |
| CLAIMS | Claim-to-evidence and failure-mode audits | `experiments/audit.py` | `full pytest suite` | implemented |

## Milestone re-audits

- Phase 0: static UE/SO, Braess, Sioux Falls, CTM conservation/source queues, and oracle certification discipline passed.
- Phase 1: held-out-demand TSTT was 160.0 versus 256.0 for dynamic shortest paths; certified-oracle relative gap was 6.31%.
- Phase 2: training and validation topology hashes were disjoint; the variable-size heterograph produced zero illegal flow on the unseen topology.
- Phase 3: OSQP projection was optimal; post-projection violation was 1.06e-7; raw violation and projection correction fell substantially after decision-focused tuning.
- Phase 4: E0–E7, 30 paired seeds, 22 robustness conditions, partial-control sweep, route assignment, statistics, metadata, and the TraCI bridge passed integration checks.
- Visualization: the stored Cambridge/MIT geographic graph imported as 20 directed CTM cells and 32 movements; the GNN/baseline trace, Folium layers, time slider, JSON summary, and checkpoint round trip passed.

## Empirical claim boundary

The checked-in smoke results validate software correctness and integration. They do not constitute the full 30-seed, held-out benchmark/SUMO study. Near-system-optimality, topology superiority, robustness, scalability, and microscopic-transfer claims must be made only after the frozen E0–E7 plan is executed on the named external benchmark data and SUMO environment.

The local gradient DSO oracle is explicitly uncertified. Only `ExhaustiveTinyOracle` outputs are labeled certified. Projection, path decomposition, and end-to-end timings are separately exposed. A real E5 transfer remains externally blocked because SUMO and a calibrated SUMO network are not installed; the adapter and fake-TraCI contract are tested.
