# Sensor convention and optional KF validation — 2026-11-13

## Scope

Pi C++14 telemetry receiver, branch experimental-deployment. STM32 firmware is
unchanged. Default heading/gyro convention is backward compatible; the candidate
mapping supported by rotation_03 is default clockwise heading with --gyro-sign 1.
Physical left-turn and positive forward encoder signs still need verification.
Optional KF adds diagnostic columns and is not wired into the NMPC state or X/Y.

## Verification

- Existing four telemetry/learning test groups plus independent gyro sign tests
  remain enabled. New native yaw KF and receiver KF tests cover wrap, known bias,
  positive covariance, missing heading in core, rejected/duplicate samples,
  invalid noise options and unchanged measured CSV columns.
- Python prototype vs C++ replay: 1529 samples; maximum yaw discrepancy
  6.67e-16 rad, maximum bias discrepancy 3.21e-17 rad/s. Replay uses explicitly
  SYNTHETIC TEL reconstructed from CSV, not the original wire capture. Missing
  raw fields are placeholders; this comparison cannot certify original CRC.
- The earlier independent-noise synthetic experiment used 10 seeds, 60 s,
  true bias 0.4 deg/s, sigma heading 0.8 deg and sigma gyro 0.35 deg/s.
  Mean yaw RMS after 20 s was 0.0784 deg; final bias 0.3746–0.4193 deg/s.
  These results establish behaviour under those assumptions, not performance
  on the correlated BNO outputs.
- Python training suite still has four pre-existing failures: the user config D
  settings test, two one-command layout tests, and candidate weight-box bounds.
  No Python training files were changed. Previous baseline comparison confirmed
  the same failures; current run log is retained with the offline artifacts.

## Next investigation performed: relative signal timing

On repository rotation_03.csv, after negating the legacy CSV gyro, compare yaw
increments with gyro integrals over approximately 100 ms, retaining rotating
samples above 3 deg/s. Scan -250..250 ms in 5 ms increments. Positive shift means
later gyro best matches an earlier heading increment.

Whole motion: best shift 0 ms, unscaled rate RMS discrepancy 3.921 deg/s.
Intervals 16–22, 22–28 and 28–30.56 s: best shifts +5, 0 and +5 ms. Data has
approximately 20 ms spacing, so this does not measure physical latency to 5 ms.
No fixed time correction is justified by this log.

Exploratory least-squares gain mapping gyro average to yaw rate is about 1.150
(over the three intervals about 1.147–1.154). With that fitted gain the whole
rate discrepancy falls to 1.640 deg/s. This gain is NOT applied to the filter,
encoder or live data and is not proven gyro scale calibration. Fusion dynamics,
heading scale/convention, timestamp clock scale and actual sensor readback need
checking against independent motion. Additive bias alone cannot generally
explain a residual proportional to angular speed.

## Follow-up on hardware

1. Confirm left turn increases normalized yaw and rate; forward wheel travel
   increases encoder total.
2. Log stationary IMU, then controlled +90/-90 degree turns and slow rotations
   against an external angle reference, retaining original TEL and host times.
3. Compare unwrapped heading change and gyro time integral for both directions
   and multiple speeds. Check STM32 time scale against host elapsed time and
   read back sensor units/mode/axis mapping. Do not fit only one trajectory.
4. Assess motor-on magnetic disturbance and calibration before selecting a
   heading noise model or consuming filter state in NMPC.

Without hardware: use replay, synthetic tests and parameter sensitivity; do not
interpret a smoother or attenuated yaw curve as improved physical accuracy.
