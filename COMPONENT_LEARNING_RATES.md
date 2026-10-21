# Component-wise Actor and Critic learning rates (controlled ablation)

## Evidence from completed TRAIN 100 episodes, gamma=.90, seed=0

The existing 100-episode run used **one** Actor SGD rate 1e-6 and
**one** Critic TD(0) rate .001. The first/last deterministic frozen
comparison shows discounted cost 14.5003897 -> 14.5001485 (only
0.0017% improvement), mean CTE 0.0200478 -> 0.0201202 m (0.36% WORSE).
Both solver success rates remain 100%; one TRAIN route does not prove
stability or generalization.

In the last episode's 1048-step log, RMS of direct-P TD feature
`delta_k*e_{k,i}**2` by state [vx,vy,psi,r,X,Y] is approximately:

```text
[1.716e-4, 8.701e-5, 7.468e-3, 7.623e-4, 6.365e-4, 6.355e-4]
```

The yaw error component's gradient is ~86x the lateral velocity
component's gradient. P_Y and P_X are also substantially larger in
*physical value*, meaning percentage/relative updates differ far more
than raw gradient magnitudes. Last-episode TD was negative on all
logged steps: do not assume a rate adjustment will remove that
bootstrap/calibration problem.

## Source configuration

Edit only `nmpc_casadi/experiment_config.py`.

```python
ACTOR_LR = 1e-6
CRITIC_LR = 1e-3

# 8 Gaussian heads: Q_vx, Q_vy, Q_psi, Q_r, Q_X, Q_Y, R_a, R_delta
ACTOR_OUTPUT_LR_FACTORS = (1., 1., 1., 1., 1., 1., 1., 1.)

# 6 Direct-P terminal weights: P_vx, P_vy, P_psi, P_r, P_X, P_Y
CRITIC_P_LR_FACTORS = (1., 1., 1., 1., 1., 1.)
```

Effective Critic TD update:

```text
alpha_P_i = CRITIC_LR * CRITIC_P_LR_FACTORS[i]
P_i <- project[P_low_i,P_high_i](P_i - alpha_P_i * TD * e_i^2)
```

Actor has shared hidden layers: per-output rates must scale the
INDIVIDUAL score-function contribution from each of the 8 Gaussian
latent coordinates BEFORE propagation through both common hidden
layers. The updated code implements exactly that. Both mean and
sigma heads are scaled per coordinate; the original diagnostic
scalar `actor_loss = -TD * log pi(z|s)` stays untouched.

Unlike the all-ones setup, this preconditioned gradient is no longer
the ordinary Euclidean gradient of the logged unweighted actor loss.
It is an **experimental diagonal per-output preconditioner** of
the Eq.(8) score-function update, not a claim that the cited paper
prescribes this weighting. No NMPC derivative, extra reward, or
state-space model gradient is introduced.

ALL-ONES reproduces the previous algorithm. Explicitly choosing
other positive finite multipliers activates a new controlled ablation.
These values are NOT auto-adjusted during the episode, do not depend
on TEST, and do not modify stage reward Qe/Re.

## How to choose components rather than guessing

1. Fix the original gamma=.90 and original reward throughout the
   comparison. Use the prior 100-episode logs as TRAIN diagnostic
   (not validation of hyperparameter performance).
2. For Critic, measure
   `RMS(TD*e_i**2) / max(P_initial_i, small_constant)` and actual
   projected per-component percentage updates; define slowly learned
   vs faster learned components in **relative units**, not just
   nominal weight magnitudes.
3. Choose cautiously capped relative multiplicative factors, keep
   the original 20%-padded P box, and run a 10-episode pilot before
   100 episodes. Check any boundary hits, extreme Actor changes,
   solver failures and CTE regressions.
4. For Actor, the proper analogue is per-coordinate RMS of the
   Gaussian score head gradients (mu AND sigma) and the output
   sensitivity `dw_j/dz_j`; a flat sampled Q/R mean is NOT a
   reliable indicator of learning due to stochastic exploration.
5. Evaluate the frozen deterministic checkpoints under identical
   TRAIN conditions and compare both the old single-rate run and
   fixed NMPC baselines. Never choose factors based on TEST.

Do NOT use inverse of a tiny measured gradient without epsilon or
clipping: near-unvisited states can make the inverse unbounded and
destroy robustness. It is also important not to increase every
weight's learning rate merely because the absolute weight is large;
observability, sensitivity and projection to the experimental
positive-Qf domain matter.

## Running another experiment

First preserve the existing completed `results/` and `images/`
**outside the active output paths** (e.g. archive to another disk
location). The new runner is designed to refuse overwriting.
Then run:

```powershell
git pull --ff-only
python .\run_experiment.py
```

No additional CLI arguments. Inspect `results/run_config.json`:
`actor_output_lr_factors` and `critic_p_lr_factors` record the actual
experiment. Plots remain `images/q_mean_episodes.png`,
`r_mean_episodes.png`, `qf_final_episodes.png` etc.

No performance improvement has been claimed merely from updating the
code. The current defaults are all neutral ones: to make a nonuniform
learning experiment, you must explicitly edit the factors.
