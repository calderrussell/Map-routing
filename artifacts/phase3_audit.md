# Phase 3 audit — projection and decision-focused training

Specification source: `specification/gnn_dso_model.tex`, lines 545–586, 672–771,
and 959–964.

| Requirement | Implementation/evidence | Status |
|---|---|---|
| Weighted nearest feasible action | CVXPY/OSQP QP in `projection/feasible.py` | Implemented |
| Nonnegativity/admissible turns | Explicit QP constraints | Pass |
| Commodity sending conservation | Outflow + nonnegative holding equals available sending | Pass |
| Sending/receiving/movement capacity | Explicit QP constraints | Pass |
| Infeasibility handling | Reported holding slack; no silent vehicle loss | Implemented |
| Differentiable layer | Masked/scaled PyTorch CTM allocator | Gradients tested |
| Projection transparency | Raw/post violation, correction distance, timing, status, slack | Implemented |
| Physics losses | Raw conservation, capacity, illegal-turn, and loop terms | Implemented |
| Decision regret | Differentiable CTM rollout relative to oracle objective | Implemented |
| Short-to-long curriculum | Horizon 1 for ten epochs, then horizon 2 | Pass |
| On-policy query policy | DAgger priority from uncertainty, correction, near-storage ratio | Implemented/tested |
| Ablations | MLP, homogeneous, no projection, no decision loss, no OD tokens, full | Registered |

Smoke result: raw violation 46.50 → 0.00061, normalized projection correction
1.428 → 0.180, post-projection violation 1.06e-7, and decision-focused training
loss 1.484 → 0.00017. OSQP reported `optimal`; the measured projection call took
approximately 4.0 ms on this tiny fixture.

Claim boundary: this proves the constraint and training mechanisms operate as designed.
Large-network projection latency and numerical conditioning remain E7 measurements.

