# First paper-aligned Actor–Critic RL training implementation

**Status as of 2026-10-21:** first end-to-end Actor → adaptive-cost NMPC →
plant → fixed paper reward → TD(0) → Gaussian policy-gradient update is
implemented and the one-transition CasADi smoke test passes. **No full
TRAIN training has been executed on GitHub**, because the pinned
`reference_split/train_xref.csv` and screening trajectory CSVs are on
the user's local machine and not present in the GitHub branch.

This file is an experiment protocol, NOT a stability certificate.

## User-approved controller structure

- Actor observation: 17 fixed physical-scale features, including vehicle
  state 6, tracking error 6, previous command 2, path progress and
  two curvature previews. Origin is relative to this route's start;
  positional scales use TRAIN length. No TEST-trained normalizers.
- Actor: 17 → 64 → 64 tanh NN, TWO learned Gaussian heads
  `mu(8), sigma(8)`, sampling 8 latent components independently
  transformed to six Q and two R admissible diagonals.
- Critic: exactly SIX learned physical P scalars, **no Critic NN**;
  `V_P(s)=-e^T P e`; `Q_f=P`.
- Candidate 2 user-approved learning start:
  `Q=[12,3,7.5,6,100,100]`,
  `R=[0.6,32]`,
  `P=[96,24,60,48,800,800]`.
- Keep original good NMPC independent, untouched:
  `Q=[12,3,15,6,183.75,192.5]`,
  `R=[0.6,16]`, `P=15Q`.
- Provisional 20%-padded Q/R/P domain from 27/27 TRAIN fixed-weight
  screening: NOT proved stable for time-varying learned weights.

## Source formulas in code

- Paper Eq. (4), fixed task reward:
  `r=-[e^T Qe e + (u_k-u_{k-1})^T Re (u_k-u_{k-1})]`
  with **constant** `Qe,Re` and the actually applied NMPC input.
- Candidate Qe/Re inherited from legacy experiment:
  `Qe=[20,20,100,20,600,600]`,
  `Re=[50,100]`. This is an *experimental evaluator candidate*
  being audited, not numerical data mandated by the paper and NOT the
  Actor-adaptive Q/R.
- Paper Eq. (6): one-step TD
  `td=r+gamma * (0 if TRUE terminal else V_P(next)) - V_P(current)`.
  Time truncation still bootstraps.
- Paper Eq. (7): `critic_loss=0.5*td^2`, direct physical projected
  semi-gradient `p_i <- project(p_i-critic_lr*td*e_i^2)`.
- Paper Eq. (8): descent `actor_loss=-td*log(pi_theta(z|s))` evaluated
  on the STORED Gaussian latent vector, with TD and latent detached.
  Both mu AND sigma heads get analytic score-function gradients,
  including shared tanh hidden layers. The actor is updated by plain
  SGD with caller-provided learning rate; NMPC is not differentiated.

**No unauthorized bonus reward, Huber loss, auxiliary neural value
network, n-step critic, TD standardization or NN gradient clipping.**
The algorithm trains online at each control step and both branches use
the SAME pre-update TD error.

## 1. Read measured reward/TD scales on TRAIN before training

The screening CSV contains only run-level summary metrics; the reward
audit instead reads `nmpc_fixed.csv` for ALL 27 full screening
trajectories and computes the candidate fixed Eq. (4) reward. With
the *explicit exploratory* gamma 0.99:

```powershell
# Run FROM the existing nmpc_casadi working directory.
git status --short
git pull --ff-only

python -m unittest -v test_rl_paper_core test_gaussian_weight_actor test_direct_p_critic test_screen_training_weights test_actor_critic_training test_actor_critic_nmpc_integration test_nmpc_core

python audit_training_reward.py `
  --screen-root .\results_screen\full_20pct `
  --output .\results_reward_audit\candidate_qe_re_g099 `
  --gamma 0.99
```

Outputs: `reward_audit.csv` containing all 27 comparisons, and
`summary.json` with reward and initial-TD ranges. An audit does
NOT train and does NOT authorize an optimizer step-size choice.
It uses the initial fixed P for TD diagnostics and consecutive
pre-step logged rows; the final transition cannot be reconstructed
as precisely as inside a live loop and is omitted from the TD audit.

Check the reward's tracking and steering contributions and the
distribution of TD before changing rates or approving 100 episodes.
`0.99` is a proposal from this project, NOT paper-prescribed.

## 2. First 10-step live end-to-end TRAIN smoke (NO full training)

Only after reviewing reward magnitudes, optionally run a SMALL
instrumented trial with explicit *illustrative* parameters:

```powershell
python train_actor_critic.py `
  --reference .\reference_split\train_xref.csv `
  --screen-manifest .\results_screen\full_20pct\manifest.json `
  --reward-audit-summary .\results_reward_audit\candidate_qe_re_g099\summary.json `
  --output .\results_rl\smoke_candidate2_seed0 `
  --acknowledge-empirical-only `
  --reward-candidate legacy `
  --gamma 0.99 `
  --sigma-lower 0.02 --sigma-initial 0.10 --sigma-upper 0.30 `
  --actor-lr 0.000001 --critic-lr 0.001 `
  --episodes 1 --steps-per-episode 10 --seed 0
```

The `gamma/sigma/actor-lr/critic-lr` values above are **diagnostic
proposals**, not approved final hyperparameters or paper values.
`--reward-candidate legacy` explicitly opts into evaluating the
old fixed `Qe,Re` candidate.
`--acknowledge-empirical-only` acknowledges that the screening set
is empirical, not a theoretical safety certificate.
The driver requires the exact SHA256 of the local TRAIN reference.
It cannot accept a held-out TEST file as the learning source.

Observe `training_history.csv`, `episode_0000.csv` (Q/R/P,
Gaussian means and sigmas, TD, reward, solver, step gradient norms)
and `checkpoint_ep0000.npz`. The checkpoint saves both Actor NN
parameters and the six physical Critic parameters with full
configuration; this initial driver has not yet implemented a
resumption or frozen evaluation mode.

For >10 steps OR >1 episode, the same-gamma full 27-case
`--reward-audit-summary` becomes MANDATORY. No 100-episode
auto-run is configured.

## 3. Interpretation and next gate

1. Confirm reward/TD magnitudes are finite and meaningful, Actor and
   Critic actually change, solver success remains acceptable and
   Q/R/P remain in the candidate box for short TRAIN interaction.
2. Approve final Qe/Re, gamma, sigma range, learning rates, optimizer
   choice (currently basic SGD), episode duration and count.
3. Add frozen checkpoint inference, independent initial conditions
   and legitimate TRAIN/validation checkpoint selection before
   final held-out TEST comparison.
4. Keep NMPC constraints and physical dynamics unchanged and separately
   document any failing solver steps without injecting unapproved
   failure reward shaping.
5. Never interpret the 27 fixed-weight screen or one live smoke test as
   proof of invariant terminal set, recursive feasibility, or stability
   throughout the 14-dimensional time-varying RL domain.
