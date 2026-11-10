# Native online Actor–Critic deployment core

Continuation explicitly authorized after Fixed NMPC parity. Implement in the
existing experimental-deployment branch; no additional approval round needed.

Port existing evaluation mathematics, not offline training. Native C++14 dense
loops preserve the 17-64-64 Gaussian Actor, both output heads, full analytic
backprop, output factors, optional mu penalty and SGD. Critic remains six direct
P coefficients with projected semi/residual TD. Preserve measured feature
scales, wrapped error, terminal mask, pre-update TD, fixed P snapshot for delayed
Actor returns, full_horizon/rollout_block scheduling, partial terminal/truncated
flush and mean batch gradients. Four independent controller copies: fixed
Q/R/Qf; actor_adapt sampled Actor/update only; critic_adapt deterministic frozen
Actor/update P; full_adapt sampled Actor and update P. P is the applied Qf.

Use standalone native math over std::vector rather than adding an NN framework
or serializing Python execution. One host-only exporter converts existing NPZ
(checkpoint config or explicit evaluation_config.json) to a versioned numeric
model, with SHA256 provenance sidecar. Explicit trained/initial_heads selection;
initial_heads retains hidden layers, resets output heads and P to nominal
values. Runtime has no Python requirement. Validate dimensions/finiteness,
bounds, LR, integer schedule and format before updates. Invalid/nonfinite
candidate updates must not mutate the live model.

For parity supply explicit epsilon streams, since NumPy PCG64 and C++ random
engines differ. Native seeded sampling is independent and reproducible within
the same C++ implementation; never promise NumPy seed parity. No new LR tuning.

Extend existing native NMPC with optional runtime Q/R/Qf parameters, leaving
fixed default and its tests unchanged. An offline four-mode runner executes
RK4 plant rollouts with supplied reference horizons and logs physical state,
Q/R/P, controls, TD, updates and final native model. It is a port validation
harness, not a replacement for route projection, odometry/state observer,
measured reward selection or hardware command adapter. No UART actuation.
Host parity tests compare all Actor tensors/gradients, critic TD/projection,
features, advantages, schedules, mode freezing, exported models, dynamic NMPC,
and short coupled rollouts against existing Python modules. Pi timing and real
sensor behavior remain untested.
