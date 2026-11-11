# Native online Actor–Critic C++14 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans; implement inline and request one fresh final review. Steps use checkbox syntax.

**Goal:** Port online adaptation and exercise four native controller modes.
**Architecture:** Validated native model, explicit proposed/observed transition
API, CasADi dynamic-weight NMPC, host NPZ export and numerical parity harness.
**Tech Stack:** C++14 standard library, CasADi 3.7.2; host NumPy unittest.
**Spec:** docs/superpowers/specs/2026-11-10-online-ac-cpp-design.md

## Global Constraints
- Preserve Python algorithms and 17-64-64 architecture; no offline training port.
- Four independent modes; no actuator output; SI state/input conventions.
- Native seed streams differ from NumPy; explicit epsilon for parity.

## Review Focus
- Nonfinite SGD candidates cannot mutate either component.
- True termination masks bootstrap; truncation does not.
- Full-horizon context samples are not reused for another Actor update.
- Frozen components and separate fixed Qf cannot change.
- Imported model sizes, bounds and positive rates are strictly validated.

### Task 1: Native Actor, Critic, features, return mathematics
Files: include/online_ac.hpp, src/online_ac.cpp,
src/actor_critic_probe.cpp, tests/test_online_ac.py.
Interfaces: Actor::propose(observation,epsilon), gradients(batch), apply;
Critic::step(error,next_error,reward,terminal); free feature/advantage functions.
- [x] Write host parity tests (forward/sampling, every gradient/SGD tensor,
  regularization/factors, TD/projection, wrapped features, n-step targets).
- [x] Run RED missing executable, then implement and run GREEN.

### Task 2: Checkpoint export and native four-mode scheduling
Files: tools/export_online_model.py, include/numeric_io.hpp,
src/online_controller.cpp, src/four_mode_probe.cpp; extend fixed_nmpc.
Interfaces: Controller::propose and observe/flush, per-mode snapshots/counters;
FixedNmpc optional dynamic weights and repeated input replacement.
- [x] Test export/roundtrip and malformed models; implement versioned model.
- [x] Test mode freezing, TD0, horizon20/full39-context schedule, terminal and
  truncated partial flush; implement coordinator.
- [x] Test coupled dynamic-weight solves and short native RK4 rollouts against
  Python under identical epsilon; implement four-mode offline runner.

### Task 3: Build, regression and documentation
Files: CMakeLists, tools/build_fixed_nmpc.py, README.md.
- [x] Build direct g++ and CMake; run new parity, fixed parity, telemetry tests.
- [x] Review implementation, fix material findings, record verification.
- [x] Document export/build/run and limits; commit/push deployment branch.

Verification record: `2026-11-10-online-ac-cpp-verification.md`. Existing Python suite has four baseline failures; no training-source changes were made.
