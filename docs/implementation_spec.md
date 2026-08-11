# Mathematical implementation map

The authoritative research design is `specification/gnn_dso_model.tex`. This note maps
its major equations to executable code.

| TeX equation | Meaning | Code |
|---|---|---|
| (2) | BPR-style link travel time | `StaticNetwork.travel_time` |
| (3) | Beckmann UE objective | `StaticNetwork.beckmann`, `frank_wolfe(..., "ue")` |
| (4–5) | SO and marginal social costs | `StaticNetwork.tstt`, `marginal_time` |
| (7–9) | Price of anarchy and remaining gap | `price_of_anarchy_gap`, `recovered_ue_so_gap` |
| (10–12) | CTM conservation, sending/receiving, capacity | `CTMSimulator.step`, `DifferentiableCTM.step` |
| (13–14) | TSTT with source queues and terminal/switch costs | CTM rollouts and DSO oracle |
| (15–16) | Receding-horizon first-action map | `RecedingHorizonOracle` |
| (17) | Masked movement softmax | `models.common.movement_softmax` |
| (18) | Learned nonnegative externality | `LearnedMarginalCostDecoder` |
| (19) | Weighted feasibility projection | `FeasibilityProjector` |
| (20) | Correlated time-varying OD demand | `dynamic_demand` |
| (21–23) | Imitation, physics, decision and total losses | `experiments.train`, `experiments.decision_training` |
| (24) | Route-share decomposition | `decompose_route_shares` |

The NumPy CTM is the independent replay/checking implementation. The PyTorch CTM is
used for gradients. QP projection uses OSQP with a Clarabel fallback and reports every
fallback/status. Tiny exhaustive enumeration is the only solver labeled certified.

