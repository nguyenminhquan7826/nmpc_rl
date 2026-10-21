# Paper-aligned Actor–Critic–NMPC rebuild

**Status:** Paper equations + **exactly six directly learned Critic P
parameters** + the NMPC cost-parameter bridge are implemented and tested.
**The Gaussian Actor, direct-P Critic, initial live Actor–Critic–NMPC TD(0) training driver, and independent/integration tests are implemented. Full TRAIN episodes and held-out TEST evaluation have NOT been run.**

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
A gated, TRAIN-SHA256-locked single-transition and short-episode
training driver with explicit SGD learning rates is now available; no long
TRAIN episode or held-out TEST evaluation has been executed. The
Critic remains a six-parameter non-NN TD(0) model.

## Source files

| File | Purpose |
|---|---|
| `nmpc_casadi/gaussian_weight_actor.py` | User-approved Actor plus 17-feature fixed physical normalization |
| `nmpc_casadi/test_gaussian_weight_actor.py` | Independent Gaussian forward/sampling/score-gradient tests |
| `nmpc_casadi/audit_critic_returns.py` | TRAIN-only finite-episode discounted-cost and Direct-P representability audit for explicit gamma choices |
| `nmpc_casadi/test_critic_return_audit.py` | Numeric regression, bounds, gamma/terminal and TRAIN-only diagnostics tests |
| `nmpc_casadi/audit_training_reward.py` | Compare old fixed reward Qe/Re candidate and pre-update initial TD on TRAIN screening trajectories |
| `nmpc_casadi/actor_critic_updates.py` | One source-faithful shared TD(0) update for Actor and six-parameter Critic |
| `nmpc_casadi/train_rl_environment.py` | Live Actor weights → adaptive-cost NMPC → plant → reward |
| `nmpc_casadi/train_actor_critic.py` | Gated TRAIN-only short/long episodes, per-step diagnostics, checkpoints |
| `nmpc_casadi/experiment_config.py` | Central gamma, train episodes, learning parameters and three filesystem roots |
| `nmpc_casadi/run_experiment.py` | One-command train, frozen comparative evaluation and figures |
| `nmpc_casadi/experiment_plots.py` | Actual X/Y path tracking, Q/R/Qf weights and learning curves |
| `nmpc_casadi/evaluate_frozen_actor_critic.py` | Reproduce deterministic untrained/frozen learned NMPC and compare to 3 fixed TRAIN screening controllers |
| `nmpc_casadi/evaluate_training_curve.py` | Replay 1,10,25,50,100-episode frozen checkpoints, compare same TRAIN conditions and fixed baselines |
| `nmpc_casadi/test_evaluate_training_curve.py` | Validate long-run checkpoint selection and strict gamma/TRAIN-only boundaries |
| `nmpc_casadi/test_frozen_actor_critic_eval.py` | Checkpoint metadata/gamma and real one-step CasADi frozen inference tests |
| `nmpc_casadi/test_actor_critic_training.py` | Test reward / TD and same TD update, terminal handling, audit gate |
| `nmpc_casadi/test_actor_critic_nmpc_integration.py` | One ACTUAL CasADi solve with Actor, direct P, paper reward and TD |
| `nmpc_casadi/training_initialization.py` | User-approved learning start: candidate 2; unchanged strong NMPC comparator; exploratory box |
| `nmpc_casadi/screen_training_weights.py` | TRAIN SHA256-locked empirical screening of representative Q/R/P weight triples |
| `nmpc_casadi/test_screen_training_weights.py` | Test start/compared weights, candidate bounds, strict TRAIN checksum and failure reporting |
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
python -m unittest -v test_gaussian_weight_actor test_screen_training_weights test_actor_critic_training test_critic_return_audit test_actor_critic_nmpc_integration test_rl_paper_core test_direct_p_critic test_nmpc_core
```

To start from another directory instead, navigate to the existing local
`D:\\GitHub\\RL_NMPC\\Code\\direct-quadratic-td-critic\\nmpc_casadi`.

## Run NMPC without RL

```powershell
python run_nmpc.py --seconds 3 --output results_nmpc\paper_rebuild_smoke
```

Outputs include CSV, config/summary JSON and diagnostic PNGs. As with the
earlier NMPC-only work, use a fresh output directory for every run.

## Approved learning-start candidate and offline screening

The user approved an initial **deliberately suboptimal but empirically
feasible** fixed NMPC:

- `Q_start=[12,3,7.5,6,100,100]`,
  `R_start=[0.6,32]`,
  `P_start=Qf_start=[96,24,60,48,800,800]`.

The ORIGINAL good fixed NMPC weights are preserved as an independent
comparator. Two uploaded TRAIN summaries showed the candidate was worse
on heading and max CTE, but better on speed; this is not a guarantee
that its common task reward is worse.

See [TRAINING_START_AND_SCREENING.md](TRAINING_START_AND_SCREENING.md)
for the explicit exploratory bounds, pinned SHA256 of
`reference_split/train_xref.csv`, Windows PowerShell commands,
3-case quick and 27-case full empirical screening, numerical limitations
and training blockers. The TRAIN CSV exists only in the user's local
working tree, not on GitHub. GitHub CI does not run those TRAIN sweeps.

## Simplified experiment — no manifest/audit arguments, just three folders

The user-requested, source-configured **one-command** workflow is now
[ONE_COMMAND_EXPERIMENT.md](ONE_COMMAND_EXPERIMENT.md):

```powershell
python run_experiment.py
```

Current `experiment_config.py` selects a **10-episode gamma=.90 isolated Critic-Qf learning-rate pilot**,
with frozen checkpoints at episodes 1, 5, 10. Edit it to change\nlearning rates, exploration sigma, checkpoint intervals and the
three paths. The runner requires ONLY the pinned existing
`reference_split/train_xref.csv` and writes flat numerical files
to `results/` and real vehicle tracking / Q-R-Qf figures to
`images/`. No external screening manifest or reward audit
file is required in this simplified mode; archived scientific
evidence and the legacy scripts remain available and unchanged.
Source-defined candidate box is *exploratory only*, not certified
for adaptive policy stability.

## Approved gamma 0.90 first episode and frozen TRAIN evaluation

The user approved `gamma=0.90`. The local Windows training run and
deterministic checkpoint evaluation are documented with exact PowerShell
commands in [RUN_GAMMA090_ONE_EPISODE.md](RUN_GAMMA090_ONE_EPISODE.md).
Following the user's request for multiple episodes, the 100-episode
learning experiment and frozen checkpoint trajectory are documented in
[TRAIN_100_EPISODES_GAMMA090.md](TRAIN_100_EPISODES_GAMMA090.md).
It uses exactly one stochastic training seed and evaluates 5 saved
checkpoints at 1,10,25,50,100 with three TRAIN initial conditions.
These runs must be executed on the user's machine with local data.
The one-episode experiment is not run in GitHub CI because the pinned
TRAIN reference and 27 fixed-screen trajectories remain on the user's
local machine. The code is CI-tested on a bundled reference for one
end-to-end real CasADi step. The inherited sigma and learning rates
remain diagnostic, not approved optimal settings.

## Before long RL training

The Actor, direct-P Critic, and one-step live CasADi integration are
implemented and tested. Candidate 2 is the approved learning start.
A 27/27 empirical PASS full TRAIN screening supports only an
**exploratory Q/R/P domain**; the paper's terminal-set and recursive
feasibility guarantees remain unproven. The old `Qe/Re`, gamma=0.99,
sigma and learning rates remain **experimental candidates**, not
paper-prescribed or fully approved.

The first 10-step TRAIN RL smoke completed with 100% solver success,
but the Critic's physical P coefficients barely moved. Before changing
rates or running longer training, use the new
[audit_critic_returns.py](nmpc_casadi/audit_critic_returns.py) to
compare 27 fixed-policy TRAIN finite discounted returns with initial
P, the best box-bounded six-P approximation, and a deliberately
non-admissible unrestricted LS diagnostic. See
[DIRECT_P_RETURN_AUDIT.md](DIRECT_P_RETURN_AUDIT.md)
for full instructions and limitations. This is an offline *batch
representability* audit, NOT an alternative TD(0) learner, and does not
substitute for frozen-policy evaluation on held-out TEST.
Paper Remark 5 still requires more than positivity/projection to
claim stability.

See [PAPER_IMPLEMENTATION_CONTRACT.md](PAPER_IMPLEMENTATION_CONTRACT.md)
and [NMPC_THEORY_AUDIT.md](NMPC_THEORY_AUDIT.md).
