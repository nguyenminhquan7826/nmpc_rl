# Fixed – Frozen – Online AC–NMPC (R3A Huber checkpoint)

The default experiment evaluates **three controllers for each physical scenario**:

1. `fixed_nmpc`: fixed nominal Q/R/Qf, no neural policy.
2. `ac_nmpc_frozen`: pretrained Actor–Critic, **same Gaussian action sampling as Online**, no network/Adam updates.
3. `ac_nmpc_online`: independently reloaded copy of **the same checkpoint** and initial RNG stream, sampled Gaussian action, network updates every `--rollout-steps` steps.

Optional `--include-frozen-deterministic` adds `ac_nmpc_frozen_deterministic`: Actor mean, no Gaussian sampling, no learning. This is a fourth controller, not a substitute for sampled Frozen in the causal comparison.

**Important:** Use the **R3A Huber best checkpoint (metadata version 11, episode 79)** with this branch (based on `experiment/critic-huber-r2a`). Do not mix it up with the separate MSE R3-A experiment, whose checkpoint metadata version differs.

## Local setup (PowerShell)

From `D:\GitHub\RL_NMPC\Code\nmpc_rl`:

```powershell
git fetch origin
git worktree add --detach ..\nmpc_rl_frozen_online origin/experiment/frozen-vs-online-ac-nmpc
cd ..\nmpc_rl_frozen_online\nmpc_casadi
python -m unittest -v test_reward_r1 test_critic_huber_r2a test_adapt_frozen
```

## First run: short nominal sanity check

```powershell
python adapt_nmpc.py `
  --checkpoint ..\..\nmpc_rl_r3a\nmpc_casadi\results_r3a\checkpoint_best_tracking.npz `
  --output eval_frozen_online_smoke `
  --nominal-only `
  --seconds 3 `
  --rollout-steps 16 `
  --actor-lr 0.001 `
  --critic-lr 0.001 `
  --seed 0
```

Update `--checkpoint` to the true checkpoint file location on your machine. Use a fresh output directory: the program rejects reuse of nonempty folders.

Inspect `comparison.csv`: exactly three distinct modes. Verify `ac_nmpc_frozen` has `updates=0` and `frozen_network_unchanged=True`; Online has at least one update at the end of a three-second rollout. Inspect `initial_sampled_pairing.csv` and expect `matched_weight_prefix=True`, `matched_cte_prefix=True` for the first 16 steps.

## Full four physical scenarios

```powershell
python adapt_nmpc.py `
  --checkpoint ..\..\nmpc_rl_r3a\nmpc_casadi\results_r3a\checkpoint_best_tracking.npz `
  --output eval_r3a_best_fixed_frozen_online `
  --scenario-set physical `
  --seconds 120 `
  --rollout-steps 16 `
  --actor-lr 0.001 `
  --critic-lr 0.001 `
  --seed 0 `
  --include-frozen-deterministic
```

Remove `--include-frozen-deterministic` for the primary three-controller comparison.

## New outputs

- `comparison.csv`: one row per controller and scenario, including `updates`.
- `pairwise_comparison.csv`: one scenario per row; retains original fixed-versus-online fields and adds frozen metrics.
- `pairwise_learning_effect.csv`: explicit Fixed–Frozen, Frozen–Online, Fixed–Online; optional mean-vs-sampled.
- `initial_sampled_pairing.csv`: audits matching action-weight and CTE prefixes **before Online's first update**.
- `*_ac_nmpc_frozen.csv`: frozen per-step logs (no update history).
- `*_ac_nmpc_online.csv`, `*_ac_updates.csv`, `*_ac_adapted_checkpoint.npz`: existing online behavior preserved.
- `nominal_frozen_vs_online_multipliers.png`: learned multiplier dynamics across Frozen and Online.

## Interpretation

The **Frozen sampled vs Online** contrast isolates the effect of continuing network updates, under the checkpoint, seed, simulator and matched sampled action noise used in this particular run. It is still an experimental comparison: trajectory divergence after updates can indirectly change observations and solver behavior. Repeat across multiple seeds, report tracking CTE **and** heading error, velocity error, steering variation, solver success and deadline misses. This experiment does not constitute a stability or real-hardware safety guarantee.
