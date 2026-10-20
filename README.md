# Paper-aligned Actor–Critic–NMPC rebuild

**Status:** Paper equations + **exactly six directly learned Critic P
parameters** + the NMPC cost-parameter bridge are implemented and tested.
**The Actor NN and full RL training loop are NOT implemented yet.**

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

## Source files

| File | Purpose |
|---|---|
| `nmpc_casadi/rl_paper_core.py` | Paper reward, TD error, loss and projection equations |
| `nmpc_casadi/direct_p_critic.py` | Exactly six directly trained positive terminal weights, TD(0) |
| `nmpc_casadi/critic_nmpc_bridge.py` | Pass Actor Q[6], R[2] and Critic P[6] to parameterized NMPC |
| `nmpc_casadi/nmpc_casadi.py` | Same NMPC dynamics/constraints/cost; optional variable Q,R,Qf parameters |
| `nmpc_casadi/run_nmpc.py` | Independent fixed-weight NMPC-only baseline (no RL) |
| `nmpc_casadi/baseline.py` | Fixed vehicle model, reference and NMPC experiments |
| `nmpc_casadi/test_rl_paper_core.py` | Tests source-defined RL mathematics |
| `nmpc_casadi/test_direct_p_critic.py` | Tests TD gradient, numerical derivative, projection, weight interface |
| `nmpc_casadi/test_nmpc_core.py` | Tests the NMPC-only baseline |

## Run the tests on Windows

```powershell
cd D:\GitHub\RL_NMPC\Code\direct-quadratic-td-critic
git status --short
git fetch origin
# After protecting any local changes / outputs, switch to the separate rebuild branch.
git switch -c experiment/paper-aligned-actor-critic --track origin/experiment/paper-aligned-actor-critic
cd nmpc_casadi
python -m pip install -r requirements.txt
python -m unittest -v test_rl_paper_core test_direct_p_critic test_nmpc_core
```

If the branch already exists locally, use
`git switch experiment/paper-aligned-actor-critic` and
`git pull --ff-only`, rather than recreating it.

## Run NMPC without RL

```powershell
python run_nmpc.py --seconds 3 --output results_nmpc\paper_rebuild_smoke
```

Outputs include CSV, config/summary JSON and diagnostic PNGs. As with the
earlier NMPC-only work, use a fresh output directory for every run.

## Before implementing the Actor and full training

The document pages 34–36 describes a stochastic Gaussian Actor with
network-produced mean AND standard deviation. However neither source
defines all numerical choices required on this 1:10 platform: observation
vector, output weight mapping, bounded admissible parameter sets, hidden
layer sizes, learning rates, and train/evaluation protocol. These must be
traced to source where possible and **explicitly approved** before writing
new training code. No changed dynamics or unverified stability claims.

Read [PAPER_IMPLEMENTATION_CONTRACT.md](PAPER_IMPLEMENTATION_CONTRACT.md)
and [NMPC_THEORY_AUDIT.md](NMPC_THEORY_AUDIT.md).
