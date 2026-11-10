# Pi 5 C++14 deployment components

The telemetry receiver is RX-only: **no motor/servo commands are transmitted**.
A separate native Fixed NMPC solver is available for offline numerical tests
(see below). The solver is not connected to telemetry or actuators yet;
Actor/Critic online adaptation has not yet been ported. It does not stop an already
moving vehicle; test receiving with the vehicle stationary and drive disabled.
No Python or CasADi runtime is needed for this receiver.

## Build on Raspberry Pi OS (Linux)

From the repository root:

```bash
mkdir -p deployment_cpp14/build
g++ -std=c++14 -O2 -Wall -Wextra -Wpedantic -Ideployment_cpp14/include deployment_cpp14/src/telemetry.cpp deployment_cpp14/src/telemetry_rx.cpp -o deployment_cpp14/build/telemetry_rx
g++ -std=c++14 -O2 -Wall -Wextra -Wpedantic -Ideployment_cpp14/include deployment_cpp14/src/telemetry.cpp deployment_cpp14/tests/test_telemetry.cpp -o deployment_cpp14/build/test_telemetry
deployment_cpp14/build/test_telemetry
```

Alternatively, with CMake installed:

```bash
cmake -S deployment_cpp14 -B deployment_cpp14/build -DCMAKE_BUILD_TYPE=Release
cmake --build deployment_cpp14/build -j2
ctest --test-dir deployment_cpp14/build --output-on-failure
```

Host serial/framing integration test (Python standard library, not needed on
the target runtime):

```bash
python3 deployment_cpp14/tests/test_rx.py deployment_cpp14/build/telemetry_rx
```

## Receive from STM32 or replay

Use the device path on your Pi (not necessarily /dev/ttyUSB0); no other
application may consume that serial stream simultaneously. User must have
serial read/configuration permissions, typically the dialout group.

```bash
deployment_cpp14/build/telemetry_rx --device /dev/ttyUSB0 --duration 30 --raw-log tel_run01.log --output pose_run01.csv --x0 0 --y0 0 --yaw0-deg 0
deployment_cpp14/build/telemetry_rx --input tel_run01.log --output pose_replay01.csv --x0 0 --y0 0 --yaw0-deg 0
```

Choose new output paths; existing files are refused. Ctrl+C closes capture.
Input is raw newline-delimited firmware TEL frames, not the Python simulation
CSV. Unsupported debug lines/bad CRCs are rejected. A partial trailing frame
is ignored. Overlong lines are discarded through the next newline.

## Exact firmware protocol

Source firmware: nguyenminhquan7826/STM32_IMU, commit
08516ddd29cdf0eb7c58e1a0f4e4339ed39f4f66, Bosch_update/Core/Src/communication.c.
USART2 115200 baud, 8 data bits, no parity, 1 stop bit, no flow control.

```
TEL,1,packet_seq,tx_ms,enc_seq,enc_ms,enc_total,enc_delta,speed_mps,steering_cmd_deg,pwm_cmd,imu_seq,imu_ms,lin_ax,lin_ay,gyro_z_dps,yaw_deg,calib,flags,crc16\r\n
```

Exactly 20 fields. Calibration is hexadecimal; flags are decimal. CRC is four
hex digits: CCITT-FALSE, init FFFF, polynomial 1021, covering TEL through flags
(exclude comma preceding CRC and CR/LF). Fields are range/finite-validated.
CRC-valid is not the same as a usable sensor sample.

Flags bits 0 and 1 must both be set. Embedded timestamp ages must be <=100 ms;
encoder/IMU timestamp skew <=50 ms. Sensor calibration bits are logged but not
used to assert physical calibration or gate integration yet. No host/STM32
clock synchronization is implied: source freshness checks cannot detect an
entire replayed/delayed serial stream. This is not an actuator safety layer.

## Odometry definition and assumptions

- Initial pose is supplied by user in a known map frame. First accepted IMU yaw
  becomes the local heading reference; absolute compass yaw is not map yaw.
- Default yaw convention matches laas-bfmc step12-status-sync: BNO heading
  increases clockwise, mathematical yaw increases counterclockwise. Use
  --imu-ccw only after verifying the sensor mounting/convention. Apply the same
  sign conversion to gyro Z and degrees->radians.
- Physical yaw is unwrapped continuously for later Actor feature parity.
- Travel = difference of cumulative signed enc_total * metres_per_count.
  encoder delta belongs only to the latest 10 ms sample and is deliberately NOT
  accumulated across 20 ms telemetry packets. Intermediate packet loss is
  recovered via cumulative counts; no distance since STM32 boot is applied to
  user initial pose.
- metres_per_count = 2*pi*.0325/(11*4*19*(38/13)) = 0.0000835635165 m.
- X/Y integrate travel with circular midpoint heading. No-slip assumption;
  no lateral velocity estimate, Kalman filter, wheel-slip/radius compensation,
  or absolute-position correction is included. vx output is STM32 speed, not
  an independent measured ground speed.
- Duplicate/reordered packets or repeated encoder samples do not add distance.
  uint32 timestamp/sequence rollover and int32 total-count rollover are handled.
- Gaps >200 ms, encoder/IMU reset/reordering, or inferred speed >0.8 m/s latch
  an odometry fault. Pose freezes until process restart with a known initial
  pose. Never silently rebase the pose after an unknown-distance gap/reset.
  Invalid flags/stale/skew samples are rejected; later data may bridge only if
  within the same gap/speed checks.

CSV: acceptance/status, source times, estimated X/Y/yaw, measured vx/yaw-rate,
signed integrated travel, cumulative ticks, flags and calibration. Rejected
samples carry the LAST estimate, not a newly valid pose; always use accepted.
No global ground-truth CTE can be inferred from this log alone.

## Validation performed

Compiled with g++ C++14 -O2 -Wall -Wextra -Wpedantic -Werror on the host.
C++ tests cover known CRC vector 123456789->29B1, valid/invalid packets, numeric
range/nonfinite rejection, stale flags, duplicates, signed count recovery,
yaw wrap, timestamp/sequence/count rollover and latched gap faults.
PTY integration verifies fragmented serial frames, raw capture, corrupt and
oversized lines, packet loss, replay, and absence of transmitted commands.
No physical Pi/STM32 test or on-device timing claim has been made. The provided
CMake file is optional; the host checks used direct g++ compilation.


## Native Fixed NMPC: offline Python/C++ parity

The C++14 library independently constructs the bicycle model, RK4 integrator,
multiple-shooting objective and constraints using CasADi C++. It does not load
a Python-exported solver. State order is `vx,vy,psi,r,X,Y`; input is
`acceleration_mps2,wheel_steering_rad`. R penalizes input **increments**.
Predicted states 1 through N-1 use Q; state N uses Qf. Heading error is wrapped.
The output acceleration is not a motor speed setpoint; there is no actuator
adapter in this milestone.

Host prerequisites: Linux, g++, Python with NumPy and **CasADi 3.7.2**.
From the repository root:

```bash
python3 deployment_cpp14/tools/build_fixed_nmpc.py
python3 deployment_cpp14/tests/test_fixed_nmpc.py
```

The helper links the Linux CasADi Python wheel's C++ headers/libraries with
`_GLIBCXX_USE_CXX11_ABI=0`, matching that wheel's GNU CMake configuration.
Python is only used to build and verify on the host; the executable itself
uses native CasADi shared libraries, including SQP/QRQP plugins. Keep the wheel
library directory available (the helper sets its rpath). This helper is not a
cross-compiler and does not produce a Pi binary on an x86 workstation.

For a native Pi build, install **aarch64 CasADi 3.7.2 C++ libraries/headers**
with SQP/QRQP and an ABI compatible with your compiler, then enable the target:

```bash
cmake -S deployment_cpp14 -B deployment_cpp14/build-native -DBUILD_FIXED_NMPC=ON -DCMAKE_BUILD_TYPE=Release -Dcasadi_DIR=/path/to/casadi/cmake
cmake --build deployment_cpp14/build-native -j2
ctest --test-dir deployment_cpp14/build-native --output-on-failure
FIXED_NMPC_PROBE="$PWD/deployment_cpp14/build-native/fixed_nmpc_probe" python3 deployment_cpp14/tests/test_fixed_nmpc.py
```

CMake gets ABI settings from `casadi::casadi`. The parity suite is a separate
Python command; CTest currently registers telemetry tests only. Leaving
`BUILD_FIXED_NMPC=OFF` preserves the CasADi-free receiver build.

`fixed_nmpc_probe input.txt output.txt` accepts a versioned numeric fixture.
`tests/test_fixed_nmpc.py::fixture` and `write_input` define the complete input:
`FIXED_NMPC_V1`, then lines `key count values...`. All physical values use SI
units. References are N+1 consecutive six-vectors; guesses contain N consecutive
six-state vectors followed by N two-input vectors. State bounds may be infinite;
states, references, inputs and model parameters must be finite. Weights must
be positive. The configured beta and lateral-speed limits are required.
Output uses `FIXED_NMPC_RESULT_V1` and the same key/count format. Dense Jacobians
are flattened column-major. Configuration is validated before constructing the
solver or opening the output file.

Exit status is 0 for a successful solve with finite objective and maximum
constraint/variable violation <=1e-6; 2 for a failed/infeasible solve; 1 for an
invalid file/configuration or I/O error. `control` and `shifted` are empty on
solve failure: never treat a failed solution as a newly valid command. Output
may contain a failed candidate `solution` for diagnostics only. This is a
numerical test tool, not a real-time controller or hardware failure policy.

Verification covers legacy parameters (.2515 m wheelbase, +/- .55 rad) and
hardware parameters (.250 m, +/-22.835397 degrees), always using identical
settings in Python and C++. It compares RK4, packed parameters, cost, objective
gradient, constraints, dense Jacobian, solved trajectory, first command and
shifted warm start, including +/-pi heading and N=1. A 20-step receding-horizon
case exercises straight/curved references and reused warm starts. Negative
tests reject malformed/nonfinite states, invalid horizon/Ts/weights and ensure
an infeasible problem produces no new control. These are parity checks, not
proof of stability, sensor estimation accuracy or Pi deadline performance.

Solver diagnostics: `solver_status_code` is 0 accepted, 1 CasADi reported
failure, 2 constraint violation above tolerance despite solver success,
3 nonfinite objective, or 4 solver exception. Failed runs also print the
CasADi return status/exception to stderr.
