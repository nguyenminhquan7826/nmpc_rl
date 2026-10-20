# Direct Quadratic TD(0) Critic — experimental branch

This branch **does not change** the legacy R3A Actor–Critic baseline. New code:

- `direct_p_critic.py`: six trainable diagonal entries `P=diag(p)`; `V_P(e)=-(e/scale)^T P(e/scale)`; direct projected semi-gradient TD(0).
- `train_direct_p.py`: new smoke/train/evaluate CLI, separately recorded logs and checkpoints.
- `test_direct_p_critic.py`: algebra, terminal mask, Actor-freeze and checkpoint tests.
- `.github/workflows/direct-p-tests.yml`: automated unit tests.

## Important interpretation

There is **no neural Critic**. `P` contains precisely six trainable coefficients. For normalized error the physical terminal diagonal passed into CasADi is `diag(P_i/scale_i^2)` so `-V_P(e)=e^T Qf e` exactly.

For batch `(e_k,r_k,e_{k+1},d_k)` all current/next values use the same frozen `P_old`:

```text
td_k = reward_k + gamma*(1-terminated_k)*V_P_old(e_next) - V_P_old(e_now)
grad_i = mean_k td_k*(e_ki/scale_i)^2
P_i_new = clip(P_i_old - critic_lr*grad_i, lower_i, upper_i)
```

The clipping box guarantees a positive diagonal but **not** Lyapunov stability or recursive feasibility. No discount factor appears in NMPC's terminal objective; TD discounted return and MPC terminal stability require separate theoretical analysis.

## Commands (PowerShell / terminal)

Run from the repository's `nmpc_casadi` directory, after installing the existing dependencies.

```powershell
python -m unittest -v test_direct_p_critic
python train_direct_p.py --mode smoke --seconds 3 --rollout-steps 1 --output direct_results/smoke
python train_direct_p.py --mode train --train-component critic --episodes 100 --rollout-steps 1 --seconds 120 --seed 0 --critic-lr 0.001 --output direct_results/critic_seed0
python train_direct_p.py --mode train --train-component both --episodes 100 --rollout-steps 1 --seconds 120 --seed 0 --critic-lr 0.001 --output direct_results/both_seed0
python train_direct_p.py --mode evaluate --checkpoint direct_results/both_seed0/checkpoint.npz --seconds 120 --output direct_results/both_eval
```

Use a **new empty output folder** for each run. This is not a continuation of the old neural-Critic checkpoint: checkpoint formats are intentionally incompatible. The initial Direct-P matrix is the original `QF_INIT`, not the pretrained R3A `Qf(s)`. Comparisons must disclose this changed initialization and train the new method independently.

For a matched controlled comparison on the **same initial Actor**, an explicit Actor-only transfer/adapter from old checkpoint must be implemented and tested before making causal claims. Do not claim it was done in this initial branch.

Default `--p-scale normalized` uses `scale=[0.1,0.06,0.2,0.5,0.1,0.1]` and maps the resulting coefficients back to physical terminal units. `--p-scale physical` runs the unnormalized theoretical baseline. Both are experimental and need separate learning-rate sweeps.

The new runner uses fixed initial pose for every episode and default one-step updates, as the first mechanistic test; a future training protocol should restore randomized training initials and held-out CAD reference testing.

## Experiments still needed

1. Run unit tests, then a 3 s smoke test with successful solver convergence.
2. Run independent seeds and compare fixed NMPC, Actor-only, Critic-only, Actor+Critic and original R3A.
3. Evaluate on held-out AutoCAD references/plant profiles using matching conditions and a new adapted evaluator; do **not** use this runner's nominal-track evaluation as held-out evidence.
4. Record mean/max/post-segment CTE, heading RMSE, speed error, reward, completion, solver reliability and P trajectories.
5. Tune the critic learning rate, scaling, update interval and bounds one at a time; do not interpret positive P alone as an NMPC stability proof.
