# Experiment D — TRAIN-only failure penalty + Actor anti-saturation

This branch's `nmpc_casadi/experiment_config.py` selects the combined
**D** variant for a new, independent 100-episode run. It reproduces the
user's completed **v3** settings, then changes **only** the training
objective. The TRAIN trajectory must be the byte-identical private
SHA256-pinned `reference_split/train_xref.csv`. No TEST file is accepted.

## Kept identical to the archived v3 run

- TRAIN initial Q = [12,3,50,6,100.75,100.75],
  R = [5,10], P = [120,30,500,60,1007.5,1007.5].
- Fixed comparator Q = [12,3,15,6,183.75,192.5],
  R = [0.6,16], P = 20*Q.
- 100 episodes, checkpoints 1/10/25/50/100, gamma=.90,
  seed=0, initial ey=+.05m and heading 0deg, box margin=.40.
- Actor LR=0.0001, Critic LR=0.0001; Actor output factors
  [1,1,1,1,2.5,2.5,1,1], Critic P factors
  [1.23,.86,.25,.41,2.5,2.5].
- Fixed original Eq.(4) task weights Qe/Re and Gaussian Actor/6P Critic.

## D changes — explicit experimental values

```python
SOLVER_FAILURE_PENALTY = 10.0
FAILURE_TERMINAL_PENALTY = 100.0
ACTOR_MU_REG_COEF = 1e-3
ACTOR_MU_REG_THRESHOLD = 2.0
```

Training-only reward is

```
r_train = r_paper
        - 10 * I[both NMPC solve attempts fail]
        - 100 * I[catastrophic terminal]
```

Normal completed-trajectory termination and time-limit truncation
do **not** incur the catastrophic terminal penalty. A recovered fallback
solve is successful and receives no solver-failure penalty.

Actor-only loss adds

```
L_actor = -TD * log pi(z|observation)
        + 0.001 * sum(max(abs(mu_i)-2, 0)**2)
```

per transition. The mu term uses analytical gradients through both
hidden layers and the Gaussian mean head, but not through the NMPC
solver. Only the TD score gradient uses the per-output LR multipliers.
The mu term is **not** multiplied by those factors. Critic TD(0) uses
the training reward; paper Eq.(4) is retained separately and unchanged.

These D penalties **change** the policy-learning objective relative to
paper Eq.(4)/Eq.(8). They are engineering hypotheses; not prescribed
optimal weights or any proof of closed-loop stability.

## Run locally after Git update

The GitHub repository has no private TRAIN file; GitHub CI tests
deterministic logic, gradient consistency and NMPC smoke, **not** 100
full episodes. On your Windows checkout, preserve all v3 outputs and
run:

```powershell
git checkout experiment/paper-aligned-actor-critic
git pull --ff-only
cd nmpc_casadi
python .\run_experiment.py
```

The new paths are `results_v3_D_penalty/` and
`images_v3_D_penalty/`. The runner refuses to overwrite populated
output folders. Do not change the private TRAIN source.

## How to evaluate

In each saved `episode_####.csv`:
`reward` is the **unmodified Eq.(4)** task reward.
`training_reward` includes the safety penalties.
`penalty_solver`, `penalty_terminal`,
`actor_mu_excess_sq`, `actor_mu_regularization` and
`actor_mu_beyond_threshold` reveal why objectives differ.
The `training_history.csv` similarly retains
`return` and `mean_reward` from original Eq.(4), and adds
`training_return`, `mean_training_reward`,
`penalty_solver_total`, `penalty_terminal_total`,
`actor_mu_regularization_mean`, and
`actor_mu_fraction_beyond_threshold`.

Frozen deterministic evaluation `comparison.csv` calls the
**unchanged** Eq.(4) reward, making it comparable to v3. Compare all
episodes' `reached_end`, the frozen checkpoint CTE, completion,
solver successes and max CTE, not solely episodic returns. A
policy that terminates early cannot win just by receiving fewer steps.

In particular, compare checkpoint 25 vs checkpoint 100; one
TRAIN seed and a single path do **not** establish TEST generalization.
Do not choose final hyperparameters or a checkpoint using TEST data.
