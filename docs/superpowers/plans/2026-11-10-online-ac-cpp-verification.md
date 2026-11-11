# Online C++14 port verification — 2026-11-10

Base: acc945b on experimental-deployment. Native host: GNU C++14 compiler 14.2,
CasADi 3.7.2; no hardware attached.

## Implementation record

Task 1: complete. Math tests first failed with missing actor_critic_probe,
then passed after the native Actor/Critic implementation. They compare every
forward output, physical weight, log probability, all eight parameter-gradient
tensors and SGD candidates, preconditioned/regularized score contributions,
Critic semi/residual gradient, projection, terminal handling, features and
multi-step returns. Shape/nonfinite/configuration cases are rejected.

Task 2: complete. Export/schedule tests first failed with missing exporter and
coordinator. Tests cover four modes, TD0, horizon20 full_horizon, rollout_block,
true-terminal and truncated short blocks; 20 combinations in total. Maximum
physical weight difference from Python: 1.14e-13. A test-fixture alias was fixed:
flattened model tensors must be copies before the Python reference mutates.

Coupled integration first failed with missing four_mode_probe. Now four short
mode rollouts and a 42-transition full_adapt/horizon20 run match Python, including
updated tensors, P, first controls, plant states and counters. Largest control
difference: 6.39e-16; physical-weight difference: 1.14e-13. Native repeated seeds
produce identical weights on the same implementation; different seeds differ.
Infeasible NMPC aborts before plant propagation or either learner update.

Contract assertions test invalid-observation RNG rollback, propose/observe
lifecycle, feedback rejection with no mutation, cancellation, end-of-run guard
and nonfinite candidate atomicity. CTest includes this test in Release builds
with assertions enabled.

Task 3: complete. Direct g++ uses -std=c++14 -O2 -Wall -Wextra -Wpedantic -Werror.
CMake builds both CasADi-enabled and core-only configurations; CTest passes.
Telemetry C++ unit and Python serial PTY/replay regressions pass. Existing Fixed
NMPC derivative/solve parity and 20-step warm-start tests remain green. The
fresh code review found no blocking findings. No deferred review findings.

An 80-step synthetic trial using the repository's
`evaluation_four_modes_5/seed_0/full_adapt/checkpoint_after.npz`, reset heads,
horizon20 and seed0 completed all four runs. Actor-only/full each perform two
Actor updates; Critic-only/full each perform 80 Critic updates. The final two
transitions are context only at truncation. Trial data remain under /tmp and
are not committed. This establishes execution, not tracking superiority.

## Existing Python suite failures

Command (pytest installed in a temporary host-only directory):

```
PYTHONPATH=/tmp/nmpc-host-checks NMPC_RL_MINIMAL_LAYOUT=1 MPLCONFIGDIR=/tmp/nmpc-mpl python3 -m pytest -q nmpc_casadi
```

Result: **124 passed, 4 failed**, plus 5 passing subtests and a pytest plugin
compatibility warning. These same four failures were reproduced from a clean
`git archive acc945b` under /tmp/nmpc-ac-baseline:

- test_component_learning_rates.py::IndependentLearningRateTests::test_user_config_d_reproduces_completed_v3_settings
- test_one_command_experiment.py::MinimalLayoutTests::test_layout_has_only_two_new_output_roots_and_pinned_source
- test_one_command_experiment.py::MinimalLayoutTests::test_single_run_command_requires_no_paths_or_manifest
- test_screen_training_weights.py::ScreeningSpecTests::test_candidate_box_contains_both_in_strict_interior

They concern existing edited configuration/layout/screening expectations. This
port does not change nmpc_casadi training/evaluation source to repair them.

## Boundaries and decisions

- The user already requested execution of the next port milestone, so the
  existing isolated deployment branch and native implementation approach were
  retained without a further approval round.
- Native RNG is not a NumPy-compatible seed stream. Explicit epsilon fixtures
  establish language parity; reproducibility across different C++ standard
  libraries is not promised.
- The numerical harness uses a short synthetic time-indexed reference and
  illustrative fixed Qe/Re. It does not port CAD projection/resampling or the
  complete Python environment. Errors/CTE from this harness are simulation
  quantities, not measured hardware truth.
- Model export/import preserves parameters and explicit configuration. Final
  models omit RNG and rollout buffers; they cannot exactly resume a live run.
- No STM32 command transmission, acceleration-to-speed adapter, lateral velocity
  observer, sensor integration or Pi latency claim is added in this milestone.
