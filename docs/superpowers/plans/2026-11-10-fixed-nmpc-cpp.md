# Fixed NMPC C++14 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox syntax for tracking.

**Goal:** Independently port Fixed NMPC and prove Python/C++ numerical parity.
**Architecture:** Native CasADi C++ model/NLP; versioned numeric files bridge
workstation Python fixtures with a standalone C++ runner.
**Tech Stack:** C++14, CasADi 3.7.2, NumPy, Python unittest.
**Spec:** docs/superpowers/specs/2026-11-10-fixed-nmpc-cpp-design.md

## Global Constraints
- State order vx,vy,psi,r,X,Y; inputs a,delta; model parameters unchanged.
- Ts .071, N 10 by default; explicit hardware and legacy profiles.
- No hardware commands, no train-code changes, no Python runtime in C++ runner.

## Review Focus
- Reject malformed/nonfinite states and wrong dimensions before solver.
- Wrapped heading differences across +/-pi.
- Exact terminal cost index and column-major decision packing.
- Infeasible solve cannot yield a valid new control.
- Hardware limits differ from legacy comparison, so compare identical configs.

### Task 1: Fixed NMPC parity contract and failing tests
Files: deployment_cpp14/tests/test_fixed_nmpc.py; tools/fixed fixture helpers.
- [x] Write parity tests against existing Python builders using synthetic
straight/curved and wrap-boundary fixtures, both config profiles.
- [x] Run with absent C++ target; expected failure identifies missing runner.

### Task 2: Native core and offline numerical runner
Files: include/fixed_nmpc.hpp, src/fixed_nmpc.cpp, src/fixed_nmpc_probe.cpp,
CMakeLists.txt. Config and results use a validated named-vector map to match the versioned numeric fixtures.
- [x] Implement dynamics/RK4, NLP, parameter packing and warm-start shifting.
- [x] Implement versioned input runner, validity checks and failure status.
- [x] Compile C++14 and run Task 1 numerical/negative tests to green.

### Task 3: Reproducible build, regression and documentation
Files: deployment_cpp14/README.md and build helper.
- [x] Run telemetry regressions and parity suite; inspect max errors/status.
- [x] Document native dependencies, ABI/version and no-hardware limitation.
- [x] Review full diff, record results, commit and push deployment branch.


## Verification record
- RED: parity suite failed because native runner was absent, before implementation.
- GREEN: four test groups, four static profiles (legacy/hardware/wrapped/N=1),
  20-step straight/curved receding-horizon test, invalid and infeasible cases.
- First-control maximum error: 3.41e-16 static; 1.12e-16 receding horizon.
- Telemetry C++ unit and Python PTY/replay regressions passed.
- Build helper uses native GNU C++14 and CasADi wheel 3.7.2. CMake native build
  and CTest checked separately; no Pi hardware/timing claim.
- Independent review found no blocker; requested solver status diagnostics
  were added as numeric failure categories and stderr CasADi return status.
