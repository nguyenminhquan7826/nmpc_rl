# Paper-aligned Actor–Critic–NMPC rebuild

**Status:** Paper equations + **exactly six directly learned Critic P
parameters** + the NMPC cost-parameter bridge are implemented and tested.
**The Actor NN is implemented and independently tested. The full RL training loop is NOT implemented yet.**

This branch intentionally builds on the NMPC-only controller in
`experiment/direct-quadratic-td-critic`; its independent NMPC baseline
remains untouched on that branch. The original older RL code is preserved at
`backup/direct-p-before-nmpc-only-reset-20261020`.

## The user-approved Critic (no neural network)

The ONLY six trainable Critic quantities are the terminal weight coefficients
`p=[p_vx,p_vy,p_psi,p_r,p_X,p_Y]`.

- `P = diag(p)`, `Qf = P`.
- `V_P(s) = -e(s).T P e(s)`, using the same six error coordinates as NMPC
  and wrapped heading error.
- Paper Eq. (6): `delta = r + gamma*V_P(s_next) - V_P(s)`.
- Paper Eq. (7): `L_c = 0.5*delta**2`.
- One-step *semi-gradient* update:
  `p_i <- project_[lower_i,upper_i](p_i - critic_lr*delta*e_i**2)`.
- The bootstrap target is held fixed in the gradient; true episode
  termination zeroes the next value, while timeout truncation bootstraps.
- Six coefficients are **physical Qf weights**, not feature-normalized
  hidden multipliers; no NN, neural baseline, regularization, Huber, n-step,
  or gradient/advantage normalization.
- The constructor requires explicit initial weights and externally
  selected positive lower/upper bounds; it does not invent any.

This is allowed by **Remark 4** of `MPC_RL.pdf`. It is deliberately
**different** from the state-conditioned neural Critic described in
`NMPC(1).docx`, pages 36–37. See
[PAPER_IMPLEMENTATION_CONTRACT.md](PAPER_IMPLEMENTATION_CONTRACT.md)
for the detailed comparison and unresolved theory issues.

## User-approved Gaussian Actor (neural network)

The 17 fixed-scale inputs are `[x(6), error(6), previous_u(2),
path_fraction, current_curvature, preview_curvature]`. The explicit
physical scales are documented in
[PAPER_IMPLEMENTATION_CONTRACT.md](PAPER_IMPLEMENTATION_CONTRACT.md).
`X,Y` are translated by the reference-path start, scaled by the TRAIN
path length; the continuous physical heading is relative to the initial
heading while heading ERROR is wrapped.

The 17→64→64 shared tanh network splits into two learned 8-output heads:
`mu(8), raw_sigma(8)`. They define an 8-dimensional Gaussian latent
policy. Each sampled latent coordinate is individually squashed to one
of the six Q or two R positive diagonal physical weights. Both heads
and all hidden layers receive score-function gradients from Eq. (8).
The deterministic initialized Actor returns the provided baseline Q/R.

**All physical Q/R bounds and Gaussian sigma limits are required
constructor arguments.** Synthetic bound values in unit tests are
illustrative only: they are NOT a certified admissible stability domain.
No training driver, gradient optimizer or test-track selection is shipped
at this stage. The Critic remains a six-parameter non-NN TD(0) model.

## Source files

| File | Purpose |
|---|---|
| `nmpc_casadi/gaussian_weight_actor.py` | User-approved Actor plus 17-feature fixed physical normalization |
| `nmpc_casadi/test_gaussian_weight_actor.py` | Independent Gaussian forward/sampling/score-gradient tests |
| `nmpc_casadi/rl_paper_core.py` | Paper reward, TD error, loss and projection equations |
| `nmpc_casadi/direct_p_critic.py` | Exactly six directly trained positive terminal weights, TD(0) |
| `nmpc_casadi/critic_nmpc_bridge.py` | Pass Actor Q[6], R[2] and Critic P[6] to parameterized NMPC |
| `nmpc_casadi/nmpc_casadi.py` | Same NMPC dynamics/constraints/cost; optional variable Q,R,Qf parameters |
| `nmpc_casadi/run_nmpc.py` | Independent fixed-weight NMPC-only baseline (no RL) |
| `nmpc_casadi/baseline.py` | Fixed vehicle model, reference and NMPC experiments |
| `nmpc_casadi/test_rl_paper_core.py` | Tests source-defined RL mathematics |
| `nmpc_casadi/test_direct_p_critic.py` | Tests TD gradient, numerical derivative, projection, weight interface |
| `nmpc_casadi/test_nmpc_core.py` | Tests the NMPC-only baseline |

## Update and run the independent tests on Windows

The user is already in the repository's `nmpc_casadi` directory
and already created the experiment branch; do NOT `cd nmpc_casadi`
a second time.

```powershell
git status --short
git pull --ff-only
python -m unittest -v test_gaussian_weight_actor test_rl_paper_core test_direct_p_critic test_nmpc_core
```

To start from another directory instead, navigate to the existing local
`D:\\GitHub\\RL_NMPC\\Code\\direct-quadratic-td-critic\\nmpc_casadi`.

## Run NMPC without RL

```powershell
python run_nmpc.py --seconds 3 --output results_nmpc\paper_rebuild_smoke
```

Outputs include CSV, config/summary JSON and diagnostic PNGs. As with the
earlier NMPC-only work, use a fresh output directory for every run.

## Before full RL training

The **Actor and direct-P Critic mathematics are implemented and tested
independently**. Actual positive parameter bounds `Q/R/P`, sigma
lower/upper and initial sigma, optimizer, learning rates, reward/evaluation
weights, offline admissibility and terminal-conditions screening, and
training/validation split decisions are still pending. Paper Remark 5
requires more than positivity/projection to claim stable closed-loop
behavior. We will not use held-out TEST to tune bounds or checkpoints.

See [PAPER_IMPLEMENTATION_CONTRACT.md](PAPER_IMPLEMENTATION_CONTRACT.md)
and [NMPC_THEORY_AUDIT.md](NMPC_THEORY_AUDIT.md).
