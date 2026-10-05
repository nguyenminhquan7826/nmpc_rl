# Actor Critic tuning for the existing NMPC simulation

Version 1 adds a stochastic Actor and a structured Critic to the supplied CAD-path NMPC. This is an experimental implementation of the paper's architecture, not an exact reproduction or a stability-certified controller. Run on a computer first. Hardware deployment is not included.

## Windows quick start

Extract the entire folder, open PowerShell inside it, then run:

```powershell
python -m pip install -r requirements.txt
python check_implementation.py
python rl_nmpc.py --mode smoke --seconds 3 --output rl_results/smoke
python rl_nmpc.py --mode train --episodes 100 --seconds 120 --seed 0 --output rl_results/seed0
python rl_nmpc.py --mode evaluate --seconds 120 --checkpoint rl_results/seed0/checkpoint.npz --output rl_results/evaluation_seed0
```

The three BAT files run smoke, training and evaluation. baseline.py is the supplied nmpc(1).py renamed to a valid Python module. Its old README/BAT filenames were stale; you do not need simulate_cad_path_qr_sweep.py for this package. To run the legacy baseline directly: `python baseline.py --mode single`.

## Architecture

- State order: vx, vy, psi, r, X, Y.
- Ts=0.02 s; N=10; restoring rear force; ZOH/RK4; SQPMethod/QRQP; same plant, constraints, virtual tail and fresh-rollout recovery as the original experiment.
- The preparation function overwrites CSV reference speed with 0.1 m/s. Evaluation stops at the real CAD endpoint, not the virtual tail. This experiment does not model stopping at the endpoint.
- Nominal Q diagonal: [12,3,15,6,183.75,192.5]; R: [0.6,16]; Qf nominal: 15 times nominal Q.
- Actor: 12 inputs -> 32 tanh units -> 3 Gaussian means. Exploration standard deviation is fixed at 0.20 in V1, not learned. Latent normal samples are transformed with tanh into bounded multipliers for position, heading and steering-increment weights.
- Actor bounds: [0.8,1.25] for each multiplier. These are conservative starting ranges, NOT a certified feasible region derived from sweep results.
- Critic: 12 inputs -> 32 tanh units -> 6 logits. Sigmoid maps each into [0.8,1.25] times its own nominal terminal weight. Critic weights are independent of the Actor's current Q. Both networks initially reproduce nominal weights.
- Critic value: V(s)=-e^T Qf(s)e. One-step semi-gradient TD updates the SAME output that supplies terminal weights. The next-state target is frozen when computing each update.
- Actor loss: -TD * log Normal(z;mu,sigma), with TD and sampled latent z treated as constants. This latent-action policy gradient is valid because the action transform is fixed. No gradient passes through NMPC.
- Adam, learning rate 1e-4 for both networks, global gradient norm limit 1. Implemented directly in NumPy; no PyTorch dependency.
- Network parameters remain fixed during each episode; state-dependent weight outputs still vary every step. One on-policy batch update is performed after each episode. Old episodes are not replayed for subsequent updates.
- Reward uses FIXED nominal Q/R: negative pre-action full-state error cost and control increment cost; -10 per final solver failure; additional -100 on failure termination or invalid state. Reward is not the adaptive solver objective.

## Observations and termination

The 12 normalized inputs are [evx,evy,epsi,er,e_parallel,e_perpendicular,vref,a_previous,delta_previous,kappa_now,kappa_horizon_end,progress_fraction]. Scales are explicit in rl_nmpc.py. Inputs are not clipped, so there is no hidden clipping change to the simulator. Current path progress and reference-manager memory are retained; this practical observation is not a formal proof of the Markov property. Generalizing to arbitrary routes may require more reference preview and solver/projection memory.

Time caps are truncations: TD still bootstraps. Route completion, invalid dynamics and three consecutive final solver failures terminate: TD does not bootstrap. Failed solves hold the previous command, as in the supplied simulator. Both solver status and constraint/variable residual <=1e-6 are required for success. A held command is not a safety-certified backup controller.

## Files and results

- rl_nmpc.py: networks, analytical backpropagation, optimizer, environment, training/evaluation CLI.
- nmpc_casadi.py: backward-compatible fixed-weight solver plus optional 14 extra NLP parameters for adaptive diagonal weights. Solver is built once per process.
- baseline.py, reference_manager.py, vehicle_model_casadi_variants.py, CSVs: original simulation helpers.
- check_implementation.py: numerical gradient check and fixed/adaptive solver parity at nominal weights.
- training_history.csv: return, tracking metrics, loss, TD and initialization per episode.
- episode_XXXX.csv: states, controls, weights, rewards, solver timings and residuals.
- checkpoint.npz: network weights only. Loading is a warm start; Adam moments and RNG state are not resumed.
- comparison.csv and comparison.png: deterministic mean-policy evaluation versus the fixed nominal baseline.
- evaluate runs eight paired scenarios: nominal; three initial-condition offsets; mass +/-10%; Iz +/-10%. The same simulator, reference and scenario apply to each pair.
- config.json: main experiment settings. Run at least seeds 0,1,2 in separate output directories before making performance claims. Training ranges are deliberately narrower than evaluation initial-condition ranges.

Comparisons use fixed rewards and physical metrics, not differently weighted NMPC objective values. completion_pct and reached_end must be checked: low CTE on an aborted episode is not success. Report all failures. Host solve time is not Pi 5 real-time evidence. Only solve/retry time is logged in V1, not the entire sensing/inference/control loop.

## Scientific limitations

This Critic structure forces V=0 at zero error, omits explicit future input-increment cost at zero error, and restricts the representable value magnitude via bounded terminal weights. Bellman fitting may plateau even when code is correct. State-dependent Qf evaluated at the current state is held constant across the NMPC horizon; it is not re-evaluated at each predicted terminal state. Critic changes also alter the effective controller between episodes, so the RL environment is nonstationary during joint learning. Do not infer policy-gradient convergence from smoke tests.

There is no invariant terminal set or terminal-decrease certification. Positive bounded weights and nominal constraints do not alone establish stability or safety under adaptation. The implementation preserves the original cost indexing (Q on predicted states 1..N-1, Qf on N, R on all input increments) for baseline parity.

100 episodes is a runnable starting experiment, not a guaranteed convergence budget. Adjust learning rates, bounds or the Critic representation only using training/validation results; keep the test set separate. Do not deploy this checkpoint on a real car before independent simulation validation and Pi/STM32 bench testing.
