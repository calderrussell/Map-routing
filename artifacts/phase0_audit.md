# Phase 0 audit — correctness before learning

Specification source: `specification/gnn_dso_model.tex`, lines 930–943.

| Requirement | Implementation | Verification | Status |
|---|---|---|---|
| Static UE and SO on Braess and Sioux Falls | `oracle/static_assignment.py`, `data_processed/networks.py` | `tests/test_static_assignment.py` | Pass |
| Analytically checkable parallel and Braess cases | Exact separable costs and Frank–Wolfe line search | UE upper-flow 1500, SO upper-flow 750; Braess UE 320000, SO 258750 | Pass |
| UE/SO normalized gap semantics | `price_of_anarchy_gap` | UE≈1, SO≈0 | Pass |
| CTM conservation | `simulators/ctm.py` | No-exit, known-exit, receiving-capacity tests | Pass |
| Oversaturated source demand remains counted | Explicit `source_queue` in state and objective | 50-vehicle demand fixture retains unadmitted vehicles | Pass |
| Small DSO checked exhaustively | `ExhaustiveTinyOracle` | Certified discrete optimum on diamond network | Pass |
| Receding-horizon DSO first action | `RecedingHorizonOracle` | NumPy replay has negligible conservation residual | Pass |
| Solver claims preserve certification status | `OracleDiagnostics` | Gradient oracle returns `locally_solved_uncertified`, no invented bound | Pass |

Command: `.venv/bin/python -m pytest tests/test_static_assignment.py tests/test_ctm.py tests/test_oracle.py -q`

Result: **8 passed**.

