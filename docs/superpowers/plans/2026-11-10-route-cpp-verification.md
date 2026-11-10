# Full-route C++14 port verification — 2026-11-10

Steps 1–2 of the user's no-hardware roadmap. Native CSV/projection/horizon/tail
and four-mode simulation are implemented; no training-source edits, sensor
noise, STM32 emulation or actuation are included.

## Data and settings

- Reference: nmpc_casadi/test_reference/test_xref.csv, length 10.7037652301 m.
  SHA256 f9e41223df41b486c00e6793cad7e4ce1b55356dcca285cd82b648740315fc6c.
- Available checkpoint: nmpc_casadi/evaluation_four_modes_5/seed_0/full_adapt/checkpoint_after.npz.
  SHA256 272db2fa18d7a3c6567df9be98ba7dea9c6e8c3a0a748c327d64d9cadf7bf482.
  This is an adapted checkpoint already in the repository, **not** a claim to
  reproduce the absent original checkpoint_best_frozen.npz's performance.
- Explicit config: nmpc_casadi/evaluation_four_modes_5/evaluation_config.json.
  SHA256 9eb14cd09a0acdb10dd297f3110ee5972490110ae6265ad5ed9b9c99c9fdafe4.
- initial_heads; independent copies of the same hidden weights; reset Q/R heads
  and P; horizon20 full_horizon; seeds 0,1,2; max2500; ey=.05m, heading offset0.
- Ts=.071, N=10; legacy .2515m wheelbase and +/-.55rad steering match Python.
  Feature length is explicitly route length, matching the existing evaluator.
- Shared saved NumPy epsilon tape, clean measurements, fixed Qe/Re from config;
  learning-only failure penalties from config are also matched.
- Host GNU C++14 / CasADi3.7.2; comparison atol1e-7, rtol1e-6, unchanged throughout.

## Full-route results

| Seed | Mode | Python steps | C++ steps | Reached end (both) | Strict parity |
| --- | --- | ---: | ---: | --- | --- |
| 0 | Fixed | 1461 | 1461 | yes | pass |
| 0 | Actor adapt | 808 | 809 | no, failure terminal | mismatch |
| 0 | Critic adapt | 1483 | 1483 | yes | pass |
| 0 | Full adapt | 1477 | 1477 | yes | pass |
| 1 | Fixed | 1461 | 1461 | yes | pass |
| 1 | Actor adapt | 1438 | 1438 | yes | pass |
| 1 | Critic adapt | 1483 | 1483 | yes | pass |
| 1 | Full adapt | 1483 | 1483 | yes | pass |
| 2 | Fixed | 1461 | 1461 | yes | pass |
| 2 | Actor adapt | 1484 | 1484 | yes | retry diagnostics differ only |
| 2 | Critic adapt | 1483 | 1483 | yes | pass |
| 2 | Full adapt | 1482 | 1482 | yes | pass |

10/12 runs meet strict parity. 11/12 meet trajectory/weights/final-model parity;
Actor seed2 differs only in primary-success/retry count at six steps. Its
maximum control difference is 9.78e-11; position difference 1.39e-12 m. For the
nine Fixed/Critic/Full runs, maximum control difference is 4.15e-14, XY difference
5.33e-15m, and P difference 4.45e-10. Metrics and final parameters are checked,
not merely plotted. The Actor seed0 incomplete rollout is not scored as a lap.

Native seed1 illustration (mean/max CTE over all logged steps):

| Mode | Mean CTE cm | Max CTE cm |
| --- | ---: | ---: |
| Fixed | 7.4845 | 15.7093 |
| Actor adapt | 5.8062 | 12.5929 |
| Critic adapt | 3.3898 | 6.6088 |
| Full adapt | 3.4357 | 6.6629 |

This table validates the port on this setup; Full does not always outperform
Critic, and these data do not establish general learning superiority.

## Actor failure investigation

At seed0, closed-loop divergence begins near step646 after tiny earlier
floating-point differences reach a sensitive SQP/QRQP status decision. We do
not hide this behind a larger tolerance or tune LR to force a match.

All ten saved failed Python requests from seed0, and all three from seed2,
were replayed with IDENTICAL numerical parameters, warm guess and bounds in the
native dynamic-weight solver. Objective, objective gradient, constraints and
constraint Jacobian matched exactly (max difference0); failure status matched
in every replay. An independent sensitivity experiment on seed0 step646 varied
one current-state component by +/-1e-14. Seven of eight nonzero perturbations
changed the solver from failure to success; the unchanged request failed.
This demonstrates numerical sensitivity around failed subproblems, rather
than claiming bitwise-identical closed-loop branch decisions across languages.

## Regressions exposed and fixed

1. Native rear-force expression originally evaluated (-Cr*error)/vx_safe;
   Python forms -Cr*(error/vx_safe). Equivalent algebra, but different AD
   roundoff mattered at failed QPs. The new exact Jacobian test failed on32
   entries by at most1.11e-16 before the grouping fix, then passed exactly.
2. Native fallback initially replaced the warm guess even after both attempts
   failed. A forced-infeasible case exposed a .007099 warm-start change. The
   retry guess is now local; failed solves preserve the existing warm guess.
3. Interpolation now uses NumPy's slope-first arithmetic, including direct
   endpoint values. A same-input exact interpolation test exposed36 differences
   up to4.44e-16 before the fix; it now passes exactly.
4. Fresh review found unchecked short input-bound vectors before CAD validation.
   Both vectors now receive shape/finiteness validation before indexing;
   zero/one-length vectors are rejected with no log created.
5. Fresh review found simultaneous reached/failure flags rejected in Fixed,
   unlike frozen_rollout. A tiny valid route with three held-input failures
   reproduces the case. Fixed now saves both flags/logs/final model, with exit2
   (failure takes precedence). Adaptive rejection is retained because the
   original compute_training_reward explicitly rejects that combination.

## Verification and boundaries

- Four reference tests: actual CSV/tail, wrapped heading/crossing/local window,
  monotonic clamp/endpoint, quoted CSV, invalid data, exact interpolation order.
- Three route tests: original Python-driver short-run parity for four modes,
  retained warm start after failed attempts, simultaneous flags and malformed
  input bounds. Tests were run RED before each corresponding fix and GREEN.
- Six Fixed NMPC tests include dynamic-weight replay and exact Jacobian checks.
- Existing online/math/export/schedule and coupled synthetic tests pass.
- Root CMake build/CTest and standard-library core-only configuration checked;
  strict direct g++ build and telemetry replay/PTy regressions checked.
- Full old Python suite retains the same four pre-existing failures documented
  in 2026-11-10-online-ac-cpp-verification.md; training source remains unchanged.
- Full outputs, epsilon and replay artifacts are under /tmp/native-route-verified-seed0,
  /tmp/native-route-verified-seed1 and /tmp/native-route-verified-seed2 on the host;
  large generated logs/models/plots are not committed.

Remaining limitation: strict Actor-only parity is not universal at failed QPs.
Use parity.json's strict and trajectory results separately, inspect incomplete
runs, and evaluate numerical solver robustness before hardware deployment.


Final verification: 19 host parity/regression test methods plus both CTest
contracts pass; telemetry PTY/replay passes. Direct g++ -Werror and root CMake
(including core-only) builds pass. A fresh final seed1 full-route rerun under
/tmp/native-route-release-seed1 passes all four strict comparisons and produces
python_cpp_overlay.png. The four unchanged old Python test failures remain
(124 passed, four failed). Fresh-review bounds and termination findings were
resolved with regression tests; no review findings were deferred. The Actor
seed0/2 strict numerical limitation is explicitly retained as experimental data.
