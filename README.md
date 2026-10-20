# Actor–Critic rebuild from the supplied paper — stage 1

This branch deliberately contains only the **source-defined mathematical
formulas** (`nmpc_casadi/rl_paper_core.py`), equation-level tests, and the
NMPC-only reference implementation. **Actor and Critic networks are NOT
created yet** because a key architecture decision (Critic NN versus six
direct coefficients) is not unambiguously settled by both user materials.
See [PAPER_IMPLEMENTATION_CONTRACT.md](PAPER_IMPLEMENTATION_CONTRACT.md).

Use `python -m unittest -v test_rl_paper_core test_nmpc_core`.
The untouched NMPC-only branch is
`experiment/direct-quadratic-td-critic`.

---

# NMPC-only baseline — rebuild from paper, without reinforcement learning

This experiment branch now contains **only the nonlinear model-predictive
controller (NMPC), six-state nonlinear bicycle model, reference processing,
fixed NMPC diagnostics, and numerical tests**. There is **no Actor, Critic,
policy, TD learning, neural network, training or RL evaluation code** in this
branch.

The previous Direct-P branch is preserved unchanged at:
[backup/direct-p-before-nmpc-only-reset-20261020](https://github.com/nguyenminhquan7826/nmpc_rl/tree/backup/direct-p-before-nmpc-only-reset-20261020).

## Fixed NMPC math (existing implementation, not modified by this cleanup)

- States: [vx, vy, psi, r, X, Y], control input [a, delta].
- Continuous bicycle dynamics: vehicle_model_casadi_variants.py.
- ZOH + RK4 discretization with Ts=0.071 s.
- Multiple shooting, prediction horizon N=10, CasADi SQP/QRQP.
- Tracking stage Q, control increment R and terminal Qf quadratic costs.
- State/input/input-increment constraints and heading angle wrapping.
- Geometric CAD XREF/reference projection by ReferenceManager.

**The code is not yet certified to reproduce the paper.** Consult
NMPC_THEORY_AUDIT.md before modifying any physical or mathematical convention.

## Setup

    cd D:\GitHub\RL_NMPC\Code\nmpc_rl_direct_p
    git pull --ff-only origin experiment/direct-quadratic-td-critic
    cd nmpc_casadi
    python -m pip install -r requirements.txt
    python -m unittest -v test_nmpc_core

## Run fixed NMPC — no training

Default CAD reference (existing smoothed reference, original baseline weights):

    python run_nmpc.py --output results_nmpc\default --seconds 120

With your two independent 10-column reference files (these are separate
**NMPC path-following tests**, not Actor/Critic TRAIN or TEST):

    python run_nmpc.py --reference .\reference_split\train_xref.csv --seconds 120 --output results_nmpc\path_one
    python run_nmpc.py --reference .\reference_split\test_xref.csv --seconds 150 --output results_nmpc\path_two

Or specify explicit fixed diagonals without any adaptive/RL tuning:

    python run_nmpc.py --q 12 3 15 6 183.75 192.5 --r 0.6 16 --qf 180 45 225 90 2756.25 2887.5 --output results_nmpc\weights_trial

The output contains nmpc_fixed.csv, summary.json, config.json and
trajectory.png, cross_track_error.png, heading_and_control.png. Each run
requires a new/empty output directory.

The long-standing pure-NMPC sweep and plant robustness modes are preserved in
baseline.py, but they have **nothing to do with Actor–Critic**:

    python baseline.py --mode single

## Scientific guardrails

1. Part I of the user's NMPC theory document is the working reference.
2. Reconcile every discrepancy in NMPC_THEORY_AUDIT.md against the **original
   paper** before claiming paper reproduction.
3. No new Actor/Critic algorithm should be added without tracing its explicit
   equations, state/action definitions and update rules back to the supplied
   paper, and verifying with the user.
4. Numerical results from the old RL experiment are preserved in the backup
   branch; they are not evidence about this newly isolated controller.
