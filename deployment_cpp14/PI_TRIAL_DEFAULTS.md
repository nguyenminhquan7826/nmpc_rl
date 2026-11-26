# Chạy từng mode với cấu hình mặc định trên Raspberry Pi 5

Đứng tại `deployment_cpp14`. Sau khi build và CTest thành công, kích hoạt
môi trường Python. Với wheel ARM64 CasADi 3.7.2 đã xác nhận ABI=1 trên Pi này,
configure bằng tùy chọn rõ ràng (không ép ABI của SDK khác):

```sh
source .venv/bin/activate
cmake -S . -B build -DBUILD_FIXED_NMPC=ON -DCMAKE_BUILD_TYPE=Release \
  -DNMPC_CASADI_ABI=1 -DNMPC_CASADI_PYTHON="$PWD/.venv/bin/python" \
  -DNMPC_RX_TEST_PYTHON="$PWD/.venv/bin/python"
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

Sau đó chuẩn bị dữ liệu **một lần**:

```sh
source .venv/bin/activate
python tools/hardware_trial.py prepare
```

Mặc định dùng checkpoint `results_v11_best/checkpoint_best_frozen.npz`,
config `../nmpc_casadi/evaluation_four_modes_5/evaluation_config.json`,
`examples/hardware_nominal.json`, khởi tạo `initial_heads`, actor horizon 20
và reference thẳng 2 m, 0.1 m/s. Quỹ đạo là mẫu kiểm tra ban đầu, không phải
quỹ đạo đánh giá đầy đủ. Đặt xe phù hợp mốc X=0, Y=0 và hướng +X khi dùng mẫu.
Các thông số plant và giới hạn giữ nguyên trong hardware_nominal.json:
m=2.400611620795107 kg, Iz=0.044 kg m², Cf=0.4, Cr=0.35 N/rad,
lf=0.130952380952381 m, lr=0.119047619047619 m, hệ số cản lăn=0.02,
g=9.81 m/s², epsilon_v=0.01 m/s. N=20, Ts=80 ms, tốc độ tối đa 0.15 m/s,
góc lái tối đa 22.835397 độ, solve budget 60 ms, state timeout 120 ms,
ACK timeout 100 ms. Đây là thông số nominal, chưa nhận dạng vòng PID thật.

Input xuất ra `hardware_trial_inputs/` và không ghi đè nếu đã tồn tại.
Có thể dùng `prepare --reference /path/to/route.csv --output another_inputs`
rồi `run fixed --inputs another_inputs`. Công cụ chuẩn bị vẫn kiểm tra
vận tốc và độ cong; không sửa quỹ đạo không hợp lệ.

## Shadow trước, motor tắt

```sh
python tools/hardware_trial.py run fixed
```

Mặc định serial /dev/ttyACM0, duration 60 s, seed 0, build/hardware_runner.
Shadow không TX và không học online. Nếu cần, thay cổng bằng
`--device /dev/ttyUSB0`, đặt `--duration 30`, hoặc chỉ rõ pose bằng
`--pose X_M Y_M YAW_DEG`. Nếu không cung cấp pose trong shadow, runner dùng
pose đầu reference; phải đặt xe tương ứng. Có thể replay với
`run fixed --input /path/to/raw_tel2.log`.

## Drive chỉ sau bench và shadow đạt

Giữ nguyên các bước an toàn, firmware, kiểm tra dấu cảm biến/servo, PID,
watchdog và ngắt nguồn motor độc lập trong HARDWARE_EXPERIMENT.md.
Chọn đúng pose thật. Ví dụ với đường thẳng mẫu và xe đặt tại gốc hướng +X:

```sh
python tools/hardware_trial.py run fixed --drive --pose 0 0 0
```

Launcher luôn thêm --keyboard khi --drive: nhấn r để bắt đầu, q để STOP và
thoát. Duration tính cả thời gian chờ. Đặt lại xe tại mốc xuất phát sau mỗi
lượt, kiểm tra đã đứng yên, sau đó chủ động chạy một trong ba mode khác:

```sh
python tools/hardware_trial.py run actor_adapt --drive --pose 0 0 0
python tools/hardware_trial.py run critic_adapt --drive --pose 0 0 0
python tools/hardware_trial.py run full_adapt --drive --pose 0 0 0
```

Các lệnh trên là lựa chọn riêng lẻ, không dán cả ba lệnh chạy liên tiếp.
Mỗi lượt dùng cùng initial.model, không tự lấy final.model của lượt trước,
và có thư mục mới trong hardware_runs/ theo mode và timestamp. Không tự
chạy mode khác khi kết thúc. Dùng --output để chỉ định thư mục mới.
Mỗi lần chỉ một process sở hữu serial. Kết quả solve timing nằm trong
control.csv và summary.txt. Chưa chứng minh đáp ứng deadline hay độ chính xác
xe thật chỉ bằng CTest.

Launcher không sửa thuật toán C++, firmware hay thông số học; Python chỉ
chuẩn bị dữ liệu và gọi binary, không tham gia vòng điều khiển.

## Kiểm chứng chu kỳ điều khiển Shadow

Live shadow dùng mốc steady_clock của Pi, chu kỳ lấy từ hardware.settings,
không đợi chênh timestamp MCU >= Ts. Mỗi bước vẫn cần mẫu IMU mới; nếu trễ,
bỏ qua slot đã quá hạn/slot quá gần, không chạy bù. Replay vẫn giữ
lập lịch theo timestamp MCU; không dùng kết quả shadow để tuyên bố Drive đã đạt 80 ms.
Kiểm tra scheduler=host_periodic_shadow, shadow_missed_slots,
shadow_lateness_max_ms trong summary.txt và host_s trong control.csv.
Giữ motor mất nguồn; chưa chạy Drive nếu chưa kiểm chứng lập lịch ACK/feedback.

## Kiểm chứng bộ lập lịch Drive (chưa xác nhận trên xe)

Drive dùng mốc host định kỳ có hiệu chỉnh pha nhỏ theo feedback, không bỏ qua
slot để chạy tiếp. Giữ 8 mẫu TEL2 và snapshot odometry tương ứng, chọn mẫu mới
nhất thỏa điều kiện ACK đúng sequence, read_start sau ACK, khoảng IMU 65..95ms
(Ts=80ms) và tuổi từ khi nhận trên host tối đa 45ms. Không tái sử dụng mẫu.
Tuổi từ lúc nhận không phải tuổi tuyệt đối của dữ liệu cảm biến.
Nếu không có mẫu hợp lệ khi quá mốc slot 10ms, dừng drive_slot_deadline;
nếu đã vượt khoảng feedback, dừng drive_feedback_interval; nếu mẫu quá tuổi,
dừng drive_feedback_stale. Không chạy bù. Hiệu chỉnh mốc tiếp theo theo
tuổi mục tiêu 10ms, tối đa +/-5ms mỗi slot (có thêm giới hạn không xếp
hai bước quá gần nhau). Ts mô hình giữ 80ms; chu kỳ host thực tế có thể
khác 80ms do lệch đồng hồ. Phải đo lại trên Pi, không giả định đúng 80ms.
Tính cả feedback/học, proposal, solver và ghi log trong ngân sách; dành 20ms
cho ghi UART trước slot kế tiếp. Giữ nguyên watchdog và cơ chế STOP.
Summary thêm scheduler=host_periodic_drive, drive_deadline_misses,
drive_lateness_max_ms, drive_compute_max_ms, feedback_count,
feedback_mean_ms, feedback_max_ms. Đây là các ngưỡng bảo thủ để kiểm thử,
không phải giới hạn đã xác nhận bằng đo lường phần cứng.

Build rồi chạy ctest --test-dir build --output-on-failure. PTY mới kiểm tra
clock MCU 39ms / host 40ms, ACK và mất feedback: không gửi RUN thứ tư,
không học từ transition chưa đủ dữ liệu. Chỉ sau khi kiểm thử đạt và hoàn
thành kiểm tra firmware, ngắt nguồn motor, servo/PID/watchdog trong
HARDWARE_EXPERIMENT.md mới xét đến drive bench. Không chạy xe trên sàn chỉ
vì Shadow hoặc PTY đã đạt.

## Native C++ CLI and profiling

After prepare, run from deployment_cpp14 with motor power disconnected:

```sh
./build/hardware_runner --mode fixed
```

Use one of fixed, actor_adapt, critic_adapt, full_adapt per trial. No Python
launcher needed. Defaults use hardware_trial_inputs, /dev/ttyACM0, seed 0,
60 seconds, unique output directory hardware_runs/<kind>_<mode>_<epoch_us>_<pid>.
Explicit --model/--settings/--reference/--output/--device/--input still work.
--help prints usage. Shorthand --drive automatically enables keyboard r/q
and still requires explicit --x0 --y0 --yaw0-deg; explicit legacy CLI behavior
is unchanged for existing test harnesses. Do not run drive on the floor yet.

control.csv appends slot_host_s (start before feedback), period_ms (live only,
first blank), reference_ms, feature_ms, actor_proposal_ms (inference, sampling,
weight mapping and controller bookkeeping), warm_start_ms, command_mapping_ms,
proposal_total_ms and feedback_ms (zero if no feedback), control_compute_ms
(start of slot to proposal end, includes feedback and transition logging but
excludes control CSV write and UART transmission). Existing solve_ms includes
set_step and evaluate_and_solve, not just the QP solve. Drive compute max still
includes control logging. transitions.csv appends feedback_ms, observe_total_ms
(transactional copy + learning), actor_update_ms (actual update; batch advantage
calculation included), critic_evaluate_ms, critic_update_ms, run_ack_wait_ms
(host time after RUN write/log to parsed matching ACK; not motor response).
No updates in shadow; unavailable update/ACK summary metrics are omitted.
Summary adds count, mean, p95 and max for each measured stage. Keep replay
period blank because it is not real-time execution. Logs are not hard-real-time
proof; profiling adds some overhead. Verify CTest and live shadow again.

### Buffered feedback and phase trim

Reproduces observed TEL2 seq 2183 -> 2185 (81 ms, valid) while latest 2186
(122 ms, invalid) was already available. The selected pose/telemetry pair is
used for both feedback and the next proposal; latest stays separate for health
and liveness. control.csv adds selected_packet_seq, latest_packet_seq,
sample_receipt_age_ms, feedback_interval_ms and drive_phase_trim_ms.
Summary adds drive_feedback_policy=buffered_phase_trim, drive_buffered_selections,
phase trim totals, sample receipt age and MCU feedback interval statistics.
New PTY regression uses host 38ms vs MCU 40ms telemetry and must complete
without invalid 120ms transitions. Old missing-feedback and safety tests remain.
No changes to firmware, reward, Ts, Q/R/P, watchdog, ACK timeout or sample
interval acceptance limits. Bench with motor and servo disconnected first.

### Active duration and confirmed shutdown

Drive --duration now measures active host time from immediately before the
first RUN, excluding initial STOP/ARM and keyboard wait. A separate
--start-timeout (default 60s, max 600s) bounds pre-RUN waiting after startup
discard. Shadow and replay duration semantics are unchanged. A duration-limited
trial can stop up to one nominal Ts early to finish the last valid transition;
route completion, q, signal or faults may stop earlier. All safety deadlines
and watchdog behavior are unchanged.

On exit from the control loop, cancel pending proposals, send STOP, and
read for at most the configured ACK timeout. Validate CRC, matching STOP
sequence, reason=0 and armed=0. No controller calls or ARM/RUN while waiting.
A missing/rejected/incorrect ACK produces nonzero exit even on duration
completion; retain original reason and record final_stop_status separately.
If STOP was not confirmed the serial destructor makes a best-effort STOP
retry (not a confirmed retry). Confirmed STOP is not resent by the destructor.
ACK confirms command guard acceptance/disarmed, NOT measured motor stop.
The independent cutoff and MCU watchdog remain required.

summary.txt adds requested_active_duration_s, start_timeout_s, active_started,
active_start_host_s, active_duration_s (excludes STOP confirmation), final_stop_seq,
final_stop_status, final_stop_confirmed, final_stop_ack_wait_ms and exit_code.
Regression covers delayed keyboard start, valid STOP ACK, missing ACK, wrong
sequence and armed ACK. Keep motor and servo supply disconnected for retest.

### Neutral watchdog integration check

After confirming the earlier encoder/PID calibration still applies, use the
short independent test described in HARDWARE_EXPERIMENT.md section 7. Build
hardware_watchdog_check and run with motor AND servo disconnected. This is
not a physical stop test; it exercises neutral CTL/ACK, silence and disarm.
