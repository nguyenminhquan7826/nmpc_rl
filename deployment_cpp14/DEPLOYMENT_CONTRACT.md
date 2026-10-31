# Raspberry Pi 5 / STM32 deployment contract

Source: train-and-evaluation at 37a6a68b744f9d1aff194254817f52421a056cb0.
Target: C++14, Linux aarch64, CasADi C++ runtime. Python may be used on the
workstation to export a trained checkpoint/solver, never required by the
on-device control loop. No offline training port is requested.

## Confirmed hardware

Pi 5 communicates with STM32 over UART/USB. Localization is encoder/IMU
odometry, not a global position camera. Encoder: JGB37-520, 11 cycles x4,
19:1 gearbox, 38/13 differential, wheel radius 0.0325 m. IMU: BNO055.

## Required behavior

Exactly four modes: fixed (Q/R/Qf init), actor_adapt (NN online updates and
sampled Q/R, fixed P), critic_adapt (fixed Q/R, online P and Qf=diag(P)),
full_adapt (both updates). The Actor needs inference AND score-function
backprop, including mean/sigma heads and shared layers. Critic is six physical
coefficients, with projection and configured TD rule. Keep the selected
Actor n-step design and its terminal/truncation semantics.

Separate physical time from control index. STM32 timestamps sensor samples;
encoder and IMU acquisition can run faster than the 71 ms controller period.
Odometry must process each encoder increment exactly once. Initial pose and
IMU heading alignment must be explicit. A no-slip vy=0 estimate is an
assumption, not an actual lateral-velocity measurement. Identify/validate a
six-state observer before claiming dynamic-state estimation parity.

NMPC produces [acceleration m/s^2, road-wheel steering angle rad]. These are
not PWM values. Confirm firmware's speed-PID/setpoint and servo calibration
before enabling actuator output. Numerical model acceleration contains
rolling/lateral terms; do not assume its first control equals measured dv/dt.

## Interface pending confirmation

Need firmware examples for RX/TX packet framing, baudrate, timestamp units,
encoder count (cumulative/increment), IMU yaw/rate units and orientation,
checksums, actuator command units/range, acknowledgements and watchdog.
No existing wire protocol or servo/PWM calibration may be inferred from PDF.
If no firmware protocol exists, define a versioned sample protocol and provide
matching STM32 requirements before a connected actuation test.

## Validation stages

1. Export artifacts, validate C++/Python Actor, gradient, Critic, reference
   preprocessing and NMPC outputs on identical numerical fixtures.
2. Replay timestamped sensor logs through C++ without connected actuators.
3. Verify serial framing, units, measured odometry and command mapping with
   wheels lifted and explicit actuator enable.
4. Time the complete loop on Pi; x86 timing is not Pi timing. Report solver
   and total-loop deadlines, fallback counts and packet age.
5. Run four physical modes with matched initial conditions and independent
   exploration seeds. Hardware odometry cannot provide ground-truth CTE:
   require external reference for actual localization/tracking accuracy, or
   label odometry-derived CTE as estimated, not true.

Do not claim hardware-tested or full C++ port complete until implemented and
validated. This document records the agreed boundary and unresolved interface.
