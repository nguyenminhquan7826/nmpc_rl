# Pi 5 C++14 telemetry receiver and encoder–IMU odometry

This milestone is RX-only: **no motor/servo commands are transmitted**. It is
not yet a C++ NMPC or Actor–Critic controller. It does not stop an already
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
