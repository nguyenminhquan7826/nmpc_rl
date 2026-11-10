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
- [ ] Write parity tests against existing Python builders using synthetic
straight/curved and wrap-boundary fixtures, both config profiles.
- [ ] Run with absent C++ target; expected failure identifies missing runner.

### Task 2: Native core and offline numerical runner
Files: include/fixed_nmpc.hpp, src/fixed_nmpc.cpp, src/fixed_nmpc_probe.cpp,
CMakeLists.txt. Config and solve results are explicit C++ structs.
- [ ] Implement dynamics/RK4, NLP, parameter packing and warm-start shifting.
- [ ] Implement versioned input runner, validity checks and failure status.
- [ ] Compile C++14 and run Task 1 numerical/negative tests to green.

### Task 3: Reproducible build, regression and documentation
Files: deployment_cpp14/README.md and build helper.
- [ ] Run telemetry regressions and parity suite; inspect max errors/status.
- [ ] Document native dependencies, ABI/version and no-hardware limitation.
- [ ] Review full diff, record results, commit and push deployment branch.
