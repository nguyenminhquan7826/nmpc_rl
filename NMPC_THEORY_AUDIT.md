# NMPC theory-to-code discrepancy register (no silent corrections)

**Scope:** the vehicle model and nonlinear MPC from Part I of the user's
NMPC(1).docx (pages 1–16); paper citations/equations must be verified
independently against the paper itself. This file is an audit, **not a claim
that the existing code exactly reproduces the paper**.

| Topic | Current NMPC implementation | Source alignment status / next action |
|---|---|---|
| State order | [vx, vy, psi, r, X, Y] | Cross-check every matrix against Part I pages 4, 9–15. |
| Model | 6-state dynamic bicycle, a and front steering delta; rear tire force convention = "restoring" by default | **PENDING**: vehicle_model_casadi_variants.py includes a distinct "paper" sign convention; resolve against authoritative paper equations and physical sign conventions before any numerical change. |
| Discretization | Zero-order hold + fourth-order Runge–Kutta, Ts=0.071 s | Structural match to Part I page 4; exact Ts is implementation-specific. |
| NMPC | Multiple shooting N=10; quadratic tracking Q, delta-input R, terminal Qf and hard constraints | Structural match to Part I pages 9–15. Check endpoint/stage indexing and all Q/R/Qf coefficients against the original paper. |
| Heading | e_psi=atan2(sin(psi-psi_ref),cos(psi-psi_ref)) | Matches wrapped-angle requirement in Part I page 9. |
| Reference geometry | CAD centerline parameterized by arc length s, curvature and reference yaw rate | Structural match to Part I pages 5–8. |
| Reference speed | Legacy baseline.make_reference uses fixed 0.1 m/s even if input speeds vary; run_nmpc --reference preserves validated supplied XREF speed within bounds | **DISCREPANCY** vs Part I pages 6–8 discussing curvature-aware speed planning plus forward/backward acceleration passes. Do not imply speed planner exists. |
| Terminal and cost conventions | Existing NMPC solver has terminal Qf only on X_N; stage Q on X_1…X_(N-1) | Check exact paper indexing before modifying. |
| Fixed Q/R/Qf | CLI defaults to baseline's established robust selected weight set | These are historical NMPC tuning values, **not** validated as paper constants. |
| RL | Completely absent from this branch | Reintroduce only after equations in paper are verified, one item at a time, separately testable. |

**Important:** The user has asked not to deviate from the paper or theory
based on personal recommendations. This reset does not change model equations,
optimizer cost or controller constraints. A **verified copy of the original
paper and any specific page/figure identifiers** are required to approve
future theory-to-code changes. If the paper differs from the user's Part I
document, document the difference and request a decision before editing.
