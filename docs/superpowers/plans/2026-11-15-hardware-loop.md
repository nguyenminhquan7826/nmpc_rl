# Hardware loop implementation plan
> Execute inline with superpowers:executing-plans, tests first.
**Goal:** Join real telemetry to four-mode C++ NMPC and guarded actuator commands.
**Architecture:** Pure C command guard on STM32; calibration/protocol and persistent
NMPC session on Pi; serial frontend defaults to shadow. Existing offline parity
runner stays unchanged.
**Tech:** C11 STM32 HAL/FreeRTOS, C++14 Linux/CasADi3.7.2, Python test tooling.
**Spec:** ../specs/2026-11-15-hardware-loop.md

## Constraints and review focus
No clocks/NDOF/KF/training edits; preserve existing branches and user commits.
Test malformed/nonfinite/out-of-range commands, timer rollover, UART overflow,
PID residual output, stale/failed sensors, missing ACK, solver deadline, shadow
silence, learning only on accepted commands and explicit hardware limits.

### Task 1: Firmware command contract and actuator gate
Files: new command_guard.h/c, actuator_task.c, tests/test_command_guard.c;
modify communication.c, encoder.c, freertos_tasks.c and host runner.
Interfaces: CommandGuard_Process(state,line,now), Tick(state,now);
Control_GetCommand(out), Control_Stop(); state snapshot owns speed/servo/armed.
- [x] Write/observe failing parser/arming/watchdog and actual PID/gate tests.
- [x] Implement strict CRC commands, ACK and bounded framing; gate actual PWM.
- [x] Run all fake-HAL checks with sanitizers and commit firmware change.

### Task 2: Pi hardware conversion and NMPC session
Files: hardware_control.hpp/cpp, hardware_session.hpp/cpp, tests.
Interfaces: command_frame/parse_ack; map acceleration+steering to speed+servo;
HardwareSession propose/commit/cancel wraps existing controller/solver/reference.
- [x] Write/observe failing conversion/protocol and session tests.
- [x] Implement with validated config, applied control and transition logging.
- [x] Run native and existing parity tests.

### Task 3: Serial/replay frontend and regression
Files: hardware_runner.cpp, CMake targets, PTY tests, example/tooling.
- [x] Write/observe failing RX-only shadow, stale/ACK/STOP and replay tests.
- [x] Implement exclusive serial owner, deadlines and signal cleanup; no automatic rearm.
- [x] Run firmware-to-Pi protocol and native solver checks; full deployment suite.

### Task 4: Review, documentation, publishing
- [x] Publish real build/run and fault/benchmark workflow; list physical checks pending.
- [x] Fresh-context review; fix material findings and rerun relevant tests.
- [x] Fast-forward user updates, commit and push existing two branches.

## Ledger
Ruling: work inline in the user's existing clean feature checkouts, preserving
requested deployment branch names; no main branch edits or new worktree needed.
Pre-flight: Task1/2 share CTL/ACK; Task2/3 share model/settings/route and session.
Physical execution/timing/AprilTag cannot be completed without user hardware.

Task 1: complete, firmware commit261e633. Twelve sanitized fake-HAL groups pass;
actual UART split/overflow and PID rapid-rearm generation covered.
Task 2: complete, conversion/session native checks and four-mode contracts pass.
Task 3: complete, default shadow no TX/no updates, seven PTY/replay/session tests,
15/15 CasADi CTest groups and13/13 dependency-free groups pass.
Task 4: reviewed independently; PID generation, serial ownership and quantization
issues reproduced and fixed. Hardware deployment instructions published with code.
Verification:17 existing Python/C++ parity tests pass; CTL->C guard->ACK and
TEL1/TEL2->receiver->audit cross checks pass. No ARM build or physical tests here.
Ruling: integer servo chosen inside both absolute and per-step steering limits,
rather than nearest unrestricted rounding followed by avoidable STOP.
Ruling: physical steps (Pi timing, bench/floor drive, AprilTag) remain pending user
hardware; software commands provided, no simulated result labeled physical.
