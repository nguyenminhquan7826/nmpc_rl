# NMPC + Actor–Critic simulation

The current code supports the original joint v7 training and a v8 experiment that trains only the Critic. See [HUONG_DAN.md](HUONG_DAN.md) for Vietnamese commands and interpretation.

## Quick start (Windows PowerShell)

Run inside `nmpc_casadi`:

```powershell
python -m pip install -r requirements.txt
python -m unittest test_critic_only -v
python rl_nmpc.py --mode smoke --train-component critic --seconds 0.4 --output smoke_v8_critic
python rl_nmpc.py --mode train --train-component critic --episodes 100 --critic-lr 0.003 --critic-nsteps 16 --rollout-steps 256 --eval-every 10 --seed 0 --initial-seed 0 --output results_v8_critic_n16
python rl_nmpc.py --mode evaluate --checkpoint results_v8_critic_n16/checkpoint_best_tracking.npz --output eval_v8_critic_n16_tracking
```

`run_train_critic.bat` and `run_evaluate_critic.bat` run the last two commands. Training prints and saves the final checkpoint comparison; use the separate evaluate command to assess the best checkpoint. A short smoke run does not reach the route's post-start metrics region and is not a performance experiment.

## Controller and networks

- Six-state dynamic bicycle: [vx, vy, psi, r, X, Y]; inputs [a, delta].
- ZOH + RK4; Ts=0.02 s, N=10; SQPMethod/QRQP with shifted warm start and fresh-rollout retry.
- The simulation imports `baseline.py`. Changes made only in the separate legacy `nmpc.py` script do not change RL.
- `baseline.py` prepares the CAD reference at 0.1 m/s, with a virtual prediction tail. Completion is measured at the real route endpoint.
- Initial diagonals in `rl_nmpc.py`: Q=Qf=[12,3,50,6,70.75,70.75], R=[0.6,70]. Change Q_INIT/R_INIT/QF_INIT for a new experiment; keep reward Q_EVAL/R_EVAL fixed for comparisons.
- Networks: 12 inputs -> 32 tanh units -> 3 Actor outputs or 7 Critic outputs.
- Actor (joint mode): Gaussian latent action, fixed sigma=0.05, tanh multipliers for common Q_X/Q_Y, Q_psi and R_delta. Other Q/R entries are fixed.
- Critic: six sigmoid multipliers in [0.5,50] for Qf and a nonnegative softplus baseline b. V(s)=-b(s)-e^T Qf(s)e. The baseline affects value estimation, not the NMPC cost.
- Current-state Qf is held fixed across the NMPC prediction horizon. This is not a neural terminal value evaluated at the predicted terminal state.
- Reward: fixed pre-action state cost plus control-increment cost, with solver/failure penalties. It does not use the adaptive NMPC objective.

## Training modes

`--train-component both` (default without a checkpoint) keeps joint learning: one-step TD for the Actor, configurable n-step target for the Critic. Both targets are frozen before updates. Parameters are updated after each rollout batch (default 256 transitions) and at the episode tail, not just once per episode.

`--train-component critic` fixes actual Q/R to Q_INIT/R_INIT, never samples Actor noise, never evaluates the Actor for control, and skips its optimizer entirely. Qf and b remain trainable. Actor parameters, Adam moments and step count remain unchanged. This experiment isolates the contribution of Critic-based Qf adaptation; Qf learning still changes the controller policy, so it is not fixed-policy value fitting.

The default Critic target length is 16; `--critic-nsteps 1` gives a one-step control. True termination removes bootstrap; time/batch truncation bootstraps using the actual available horizon. Training TD RMS uses this n-step target; eval TD RMS always uses a one-step target.

Use `--initial-seed 0` in BOTH new runs for matched episode offsets independent of Actor noise. Critic mode defaults this seed to `--seed`; joint mode without this option retains the legacy v7 RNG sequence. Previous joint v7 results therefore do not have the same per-episode offset schedule as a new critic-only run, even with the same `--seed`. Deterministic eval scenarios are the same.

## Checkpoints and evaluation

- Without explicit CLI overrides, mode, n-step length and initial seed are inferred from checkpoint metadata.
- Legacy v7 joint checkpoints remain usable in joint mode. Critic-only checkpoints use version 8 and require matching mode/configuration.
- Start the isolated Critic experiment fresh; do not convert an already jointly trained v7 network into this experiment.
- Checkpoints save parameters, Adam state, exploration RNG and (when enabled) initial-condition RNG. Resuming in a new output folder starts a new history/selection run; this is not a full training-session resume.
- `checkpoint.npz`: final; `checkpoint_best_tracking.npz` and `checkpoint_best_return.npz`: selected on nominal validation.
- Selection prioritizes feasibility, then completion/solver health, then CTE or return. Best does not imply feasible.
- Joint eval: 5 controllers × 8 scenarios. Critic-only eval: baseline, fixed_qf and critic_only × 8 scenarios.
- `fixed_qf` uses Q_INIT/R_INIT and 50*QF_INIT. `critic_only` uses Q_INIT/R_INIT and learned state-dependent Qf.
- Value diagnostics are exported only for the policy that was trained, including critic_only for the new mode.
- Compare return, mean/max post CTE, heading/speed RMSE, feasibility and completion; do not use value loss alone to claim improved control.
- Report host timing separately from Pi 5 timing. Hardware deployment is not implemented.

## Files to review

`training_history.csv`, `update_history.csv`, `validation_history.csv`, `best_checkpoints.json`, `config.json`, and the separate best-checkpoint eval folder. The latter contains `comparison.csv`, `ablation_deltas.csv`, per-controller trajectories and Critic diagnostics. Episode CSVs log actual applied weights and whether Actor exploration was enabled.

To inspect learning, check actor_updated=0 and actor_parameter_delta=0 in critic mode, constant Q/R in episode logs, changing Qf, value-vs-return errors and deterministic controller comparisons. The episode weight plot mixes state-dependent outputs and learning; joint mode also includes exploration.

## Validation

`test_critic_only.py` checks frozen Actor parameters/Adam/noise, checkpoint compatibility and RNG restoration, paired initialization, controller routing and terminal/bootstrap targets. Integration runs should exercise train -> best-checkpoint eval, including the constant applied Q/R and exported diagnostics. Smoke tests do not establish convergence or tracking improvement.
