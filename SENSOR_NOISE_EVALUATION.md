# Optional Gaussian sensor-noise experiment (four Actor–Critic modes)

This experiment perturbs **measurements**, not the vehicle dynamics:
`x_measured[k] = x_true[k] + N(0, diag(sigma**2))` for
`[vx (m/s), vy (m/s), psi (rad), yaw rate (rad/s), X (m), Y (m)]`.

The baseline remains **noise-free by default**. Enable noise only with
`--sensor-noise`. Example standard deviations, **not calibrated sensor
specifications**, are `[0.005, 0.005, 0.00872664626, 0.01, 0.005, 0.005]`.
Noise is additive, iid across control steps and components, and sampled once
per control step. No IMU bias, delay, quantization, or temporal correlation is
modeled here; the separate `sensor_models.py` encoder/IMU simulation is **not**
yet wired to this evaluation. This test deliberately isolates white noise.

## Data flow

- The plant evolves only from `x_true` and the actual applied input.
- NMPC's initial state, path projection, Actor features and Critic TD errors
  use `x_measured`. Previous actuator command remains the known applied input.
- The online training reward is computed using noisy observation error.
- Logged/evaluation `reward`, CTE, progress, heading RMSE and end-of-route
  detection use an **independent true-state path projector** and true plant
  error. The controller cannot use these ground-truth metrics.
- CSV contains both `reward_true`/`reward_observed`,
  `cte_m`/`cte_measured_m`, `state_true_0..5`,
  `state_measured_0..5` and `sensor_noise_0..5`.
  Use `reward` (ground truth) for comparing controller performance, and
  `training_reward` (observed, possibly penalty-adjusted) for analyzing TD.
- Separate sensor RNG never consumes Actor's exploration RNG.
  Noise is keyed by `(sensor-seed + evaluation-seed, control-step)` and is
  identical across the four controller modes at the same time index,
  irrespective of the vehicles' diverging trajectories.
  Modes with unequal run lengths only share their overlapping time indices.

## Running

From `nmpc_casadi/` with the same checkpoint and evaluation route:

```powershell
python evaluate_controller_modes.py `
  --checkpoint "PATH_TO_CHECKPOINT.npz" `
  --reference "PATH_TO_REFERENCE.csv" `
  --output "results_sensor_clean" `
  --max-steps 2000 --seeds 0 1 2 --save-data
```

With **illustrative** Gaussian noise:

```powershell
python evaluate_controller_modes.py `
  --checkpoint "PATH_TO_CHECKPOINT.npz" `
  --reference "PATH_TO_REFERENCE.csv" `
  --output "results_sensor_noise" `
  --max-steps 2000 --seeds 0 1 2 --save-data `
  --sensor-noise --sensor-seed 10000
```

Optional custom physical-unit standard deviations:

```powershell
  --sensor-noise --sensor-std 0.005 0.005 0.00872664626 0.01 0.005 0.005
```

Outputs must be new or empty directories. Each seed contains the four modes:
`fixed`, `actor_adapt`, `critic_adapt`, and `full_adapt`.
The manifest `evaluation_config.json` preserves all sensor parameters and
the reference/checkpoint hashes. `comparison.csv` and `summary_by_mode.csv`
report true-state performance. Use the *same checkpoint, reference, seed,
optimization settings and condition* for clean-vs-noisy comparisons.

## Interpreting results

Compare complete/feasible rollouts first, then **true-state** Mean/Max CTE,
heading RMSE, sum of true-state reward, solver-success rate and timing.
Inspect any additional solver failures introduced by noisy measurements.
These runs allow online adaptation on the evaluation route. They are therefore
**not** held-out frozen-policy generalization evidence, and finite simulations
do not establish closed-loop stability of adaptive terminal weights.
Never claim the illustrative noise levels describe real sensor accuracy.

The original TRAIN runner does not enable sensor noise automatically.
