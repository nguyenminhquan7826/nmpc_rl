# Hardware loop specification
User authorization: implement the ordered steps agreed in chat; proceed inline.
Existing firmware/Pi feature branches remain the deployment targets. No clock,
NDOF, Kalman or training changes. AprilTag hardware validation remains a later
physical task. The following are implementation decisions, not measured results.

- Add CRC protected sequenced ARM/RUN/STOP commands and ACK. Default disarmed;
  300 STM32 ms watchdog; forward only 0..0.5 m/s; servo45..105 centered75.
  Legacy CMD is no longer an actuator path. Oversized/overflowed lines discarded.
- Keep PID state owned by encoder task. Disarmed/zero request clears PID and PWM;
  actuator independently gates on command timeout and fresh encoder/healthy IMU.
- Reuse laas-bfmc cubic calibration (75 center, +/-30 servo travel,
  equivalent bicycle angle +/-22.835397 degrees), record source provenance.
  Acceleration maps to next speed request from measured vx over Ts; no reverse.
- C++ hardware runner supports TEL replay and live serial; default shadow emits
  no bytes and does not learn from unapplied commands. --drive is explicit,
  requires ACK, source/host freshness, TEL2 fusion5/0, nominal step and solve
  deadlines, bounded commands and STOP on exit/fault. No automatic rearm.
- State uses raw BNO odometry, explicit gyro sign +1 and explicit vy=0 assumption.
  Hardware geometry/limits in settings; reject incompatible configuration.
- Four existing modes reuse OnlineController. Live learning only after accepted
  command/next fresh state; shadow proposes/cancels without learning. Failures
  cancel, stop, do not hold-and-learn. Logging records inputs/weights/commands,
  ack, timing, state, reward/update and exit reason. Save initial/final model.
- Document on-Pi timing, motor-disabled bench tests, fixed low-speed first,
  subsequent four-mode experiment and camera validation. No physical test claims.
