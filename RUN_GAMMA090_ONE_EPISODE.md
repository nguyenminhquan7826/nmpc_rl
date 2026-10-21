# FIRST full-length TRAIN episode: gamma=0.90 and frozen evaluation

**Approval:** The user chose `gamma=0.90` on 2026-10-21.
The previous ten-step trial's diagnostic settings are REUSED for this
first exploratory run, NOT assumed optimal or paper-prescribed:
`sigma=[.02,.10,.30]`, Actor SGD `1e-6`, Critic projected
semi-gradient TD(0) `1e-3`, seed 0.

**Scope:** One (1) stochastic Actor–Critic–NMPC episode on pinned TRAIN,
followed by frozen deterministic Actor inference on the very same TRAIN.
Compare frozen post-training controller against frozen untrained
Actor–Critic plus THREE strong/weak fixed NMPC comparators from
previous TRAIN screening. No final TEST evaluation; this is a pilot,
not a scientific proof of improved generalization.

Do NOT overwrite previous gamma=.99 reward audit or smoke.

## Prerequisites

Working dir is ALREADY
`D:\GitHub\RL_NMPC\Code\direct-quadratic-td-critic\nmpc_casadi`
on the user's Windows machine. Do not `cd` again. Full screening
27/27 case logs (including per-case `nmpc_fixed.csv`) already exist
in `results_screen\full_20pct`. GitHub Actions checks code with
bundled engineering reference, NOT the user's local pinned TRAIN.
The user, not the remote Desktop Commander ts-03, has this local
D:\ project path.

Before running, ensure no other training process holds the GPU/CPU
or modifies the same project output paths.

## A. Audit the fixed reward at newly chosen gamma

The already computed .99 reward audit CANNOT be reused at .90,
because its TD diagnostics reflect the old gamma. Reward Qe/Re
remain the same; gamma is the only changed parameter here.

```powershell
git status --short
git pull --ff-only

python audit_training_reward.py `
  --screen-root .\results_screen\full_20pct `
  --output .\results_reward_audit\candidate_qe_re_g090 `
  --gamma 0.90
```

Expected audit JSON: `cases=27`, pinned SHA256
`afa67ac24ef61b591b1a16498bcb1ee5e521452cf0f8036de00923d062940b98`,
`gamma_diagnostic=0.9`.

## B. Exactly ONE TRAIN learning episode

The full fixed-weight runs on TRAIN completed in ~1006–1050
steps. Use max 1150 steps here to permit a comparable full route.
The driver stops early if the reference end is reached; 1150 is
a cap, not the intended actual step count.

```powershell
python train_actor_critic.py `
  --reference .\reference_split\train_xref.csv `
  --screen-manifest .\results_screen\full_20pct\manifest.json `
  --reward-audit-summary .\results_reward_audit\candidate_qe_re_g090\summary.json `
  --output .\results_rl\gamma090_one_episode_seed0 `
  --acknowledge-empirical-only `
  --reward-candidate legacy `
  --gamma 0.90 `
  --sigma-lower 0.02 --sigma-initial 0.10 --sigma-upper 0.30 `
  --actor-lr 0.000001 --critic-lr 0.001 `
  --episodes 1 --steps-per-episode 1150 --seed 0 `
  --ey 0.05 --heading-deg 0
```

Driver outputs:

- `training_history.csv` — one-line learning summary
- `episode_0000.csv` — every sampled Q/R, P before/after,
  Gaussian heads/sigma, TD, reward, solver progress and wall time
- `checkpoint_ep0000.npz` — checkpoint with 6352 Actor parameters,
  six directly learned Critic P and exact run metadata
- `run_config.json`, and unmodified reference copy/tail

If `reached_end=false` or `failure_terminal=true`, or solver
success drops markedly, STOP. Do not infer controller performance
from a truncated or failed rollout; share logs for review.

## C. Frozen deterministic evaluation on the same TRAIN

No exploration/sampling or gradient updates. Evaluate both
`untrained_deterministic` and `trained_deterministic`
with same original offsets +0.05m/0deg, same NMPC solver/plant and
the same fixed task reward Eq. (4).

Read and compare three prior fixed-NMPC logs for EXACT same TRAIN
and initial offsets:
- `learning_initial`: weak candidate 2
- `strong_fixed_baseline`: original strong comparator unchanged
- `stage_lower_terminal_upper`: extra high-performing fixed
  comparator identified during TRAIN screening

```powershell
python evaluate_frozen_actor_critic.py `
  --reference .\reference_split\train_xref.csv `
  --screen-root .\results_screen\full_20pct `
  --screen-manifest .\results_screen\full_20pct\manifest.json `
  --checkpoint .\results_rl\gamma090_one_episode_seed0\checkpoint_ep0000.npz `
  --output .\results_rl\gamma090_frozen_eval_seed0 `
  --max-steps 1150
```

Optional sensitivity check across ALL THREE previously screened TRAIN
initial conditions (two conditions have NOT been used to update
Actor/Critic in this one-episode pilot; still not TEST):

```powershell
python evaluate_frozen_actor_critic.py `
  --reference .\reference_split\train_xref.csv `
  --screen-root .\results_screen\full_20pct `
  --screen-manifest .\results_screen\full_20pct\manifest.json `
  --checkpoint .\results_rl\gamma090_one_episode_seed0\checkpoint_ep0000.npz `
  --output .\results_rl\gamma090_frozen_three_conditions_seed0 `
  --max-steps 1150 --all-train-conditions
```

## Evaluation outputs and decision gate

`comparison.csv`: five controllers x one (or three)
initial conditions. Metrics calculated consistently from pre-step
states, with evaluation after progress >= 1.3 m:

- `discounted_cost_from_start` with **fixed Eq. (4) reward**
  and gamma=.90; smaller cost means higher return.
- `mean_cte_post_m`, `max_cte_post_m`,
  `rmse_heading_post_rad`, `rmse_vx_post_mps`.
- `solver_success_pct`, `reached_end`,
  `truncated`, `failure_terminal`, completion percentage.
- solve time P95 (if available), useful for embedded budgets.

`eval_config.json` describes the run, frozen checkpoint,
TRAIN isolation and limitations.
`trained_deterministic_ey+0.050_hdg+0.0.csv` and
`untrained_deterministic_ey+0.050_hdg+0.0.csv` give every step
for checkable replay, including Q/R/P.

**A positive learning claim requires**: checkpoint frozen performance
beats the appropriate comparator(s) under the SAME evaluation
protocol and reliable solver, not just a stochastic TRAIN return
improvement. The one-episode pilot permits only an exploratory
feasibility statement; validate multiple seeds and finally use
a truly held-out TEST set AFTER hyperparameters/checkpoint are fixed.

**Important scientific caveat:** the initial 20%-padded P bound
`[76.8,...,640]` to `[216,...,3465]` remains artificially
narrow compared with some offline positive-P return fits. One
episode with this provisional box may show very little Critic
learning. This is not an argument to directly put huge offline
fitted P into terminal NMPC without renewed feasibility and
theoretical terminal-set checks.

## What to send back for quantitative review

Upload these from the new run:
1. `results_rl\gamma090_one_episode_seed0\training_history.csv`
2. `results_rl\gamma090_one_episode_seed0\episode_0000.csv`
3. `results_rl\gamma090_frozen_eval_seed0\comparison.csv`
4. `results_rl\gamma090_frozen_eval_seed0\eval_config.json`

Add the `checkpoint_ep0000.npz` if you want a thorough
parameter-by-parameter audit. Keep the original `run_config.json`
and local TRAIN reference unchanged.
