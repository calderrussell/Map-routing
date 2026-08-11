# Phase 2 audit — heterogeneous topology-general pipeline

Specification source: `specification/gnn_dso_model.tex`, lines 427–543 and 952–957.

| Requirement | Implementation/evidence | Status |
|---|---|---|
| Cell, OD, movement, and global entity context | `models/features.py`, `models/heterogeneous.py` | Implemented |
| Relation-specific message passing | Directional cell, OD→origin, destination→OD, origin→OD, and global relations | Implemented |
| Temporal state | Separate cell and OD GRU cells | Implemented |
| Physical normalization | Occupancy/storage, flow/capacity, local capacity/free-time scales; no test-topology z-score | Implemented |
| Commodity scaling | Active-token sparsification and destination batching | Implemented/tested |
| Synthetic graph families | Grids and connected random-geometric planar graphs | Implemented |
| Benchmark network path | TNTP loader and link-to-cell converter; Sioux Falls creates 76 road cells | Implemented/tested |
| Split before scenarios | `TopologySplits` plus fit/tune/final-test authorization | Implemented/tested |
| Variable topology execution | Train on diamond + 2×2; validate on distinct 2×3 hash | Pass |
| Admissible output | Zero flow on illegal/unreachable validation movements | Pass |

Smoke metrics: training imitation loss 0.01314 → 0.00572; held-out-topology
imitation loss 0.01031. Eight training labels were exhaustively certified and four were
retained as explicitly uncertified local CTM-oracle solutions.

Claim boundary: this verifies architecture portability, split integrity, and output
feasibility masks. It does not establish that the heterograph outperforms a homogeneous
GCN on test-network system cost; the frozen E3 experiment is required for that claim.

