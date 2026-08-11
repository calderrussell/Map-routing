# Phase 1 audit — same-topology surrogate

Specification source: `specification/gnn_dso_model.tex`, lines 945–950.

The reproducible smoke experiment is implemented by `experiments/phase1.py`; its raw
result is generated at `artifacts/runs/phase1_smoke.json`.

| Requirement | Evidence | Result |
|---|---|---|
| Destination-conditioned GCN/GRU | `models/homogeneous.py` | Implemented |
| Movement logits and masked softmax | `ModelOutput.logits`, `movement_softmax` | Illegal/unreachable turns receive zero flow |
| Exact CTM allocator | `simulators/ctm.py` | Used in closed-loop training evaluation |
| First-action imitation | 32 certified tiny-oracle demonstrations | Loss 0.00546 → 0.00141 |
| Held-out demand | Fixed profile not drawn from training generator | Evaluated |
| Improve over dynamic shortest paths | TSTT 160.0 versus 256.0 | 37.5% lower |
| Oracle-relative quality | Certified discrete oracle TSTT 150.5 | 6.31% relative gap |
| Physical correctness | Maximum conservation residual | 3.55e-15 |

The model inference time was 3.10 ms for the eight closed-loop decisions; the tiny
certified enumeration took 3.00 s. These timings are diagnostic only and do not support
a scalability claim.

Claim boundary: this phase validates the pipeline on one tiny topology. It is not
evidence of topology generalization, real-network performance, or SUMO transfer.

