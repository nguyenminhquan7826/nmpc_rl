# Vòng thực nghiệm C++14: STM32 ↔ Pi 5

Đã có chương trình ghép telemetry thật → odometry → reference → NMPC → bộ
chuyển đổi lệnh và giao thức STM32. **Mặc định là shadow: chỉ nhận, tính và ghi
log; không gửi lệnh, không học từ các lệnh chưa áp dụng.** `--drive` là lựa chọn
riêng để chạy đóng vòng sau các bài kiểm tra bàn. Chưa có kết quả chạy xe/Pi
hoặc AprilTag cho phiên bản tích hợp này.

## Các giới hạn cần hiểu

- Dùng yaw BNO NDOF và gyro dấu `+1`, chưa dùng Kalman bổ sung; vy luôn bằng 0,
  là giả thiết chạy chậm, không phải phép đo vận tốc ngang.
- Encoder geometry giữ nguyên. Odometry không được camera sửa vị trí.
- Tọa độ xe ban đầu phải đúng với bản đồ/đường CSV; yaw BNO mẫu đầu chỉ tạo
  mốc tương đối, không tự xác định hướng của bản đồ.
- Cấu hình mẫu: wheelbase0.25m, tốc độ tối đa0.15m/s, Ts80ms, góc xe tối đa
 22.835397°. Các tham số động lực học là nominal, chưa nhận dạng vòng PID thực.
- NMPC sinh a; `v_request = vx + (a - rolling_acceleration)*Ts`. Chuyển đổi này
  giả thiết PID tốc độ bám yêu cầu đủ nhanh; không tương đương áp gia tốc vật lý
  tức thời. Gia tốc thực và đáp ứng inner-loop vẫn cần đo. Hạng lực ngang không
  nằm trong bộ chuyển đổi đơn giản này. Chưa khẳng định tương đương plant mô phỏng.
- Góc servo là lệnh, không có cảm biến đo góc bánh lái thực. Dùng cubic đo trong
  `laas-bfmc/src/laas_core/SteeringCalibration.hpp`: A0.9666368131,
  B−0.0002282854502; center75°, servo45..105°. Chọn integer servo gần nhất còn
  nằm trong giới hạn góc cấu hình; dấu servo phải kiểm tra trên xe.
- ACK xác nhận firmware nhận/setpoint, **không chứng minh motor đã đạt vận tốc**.
  Transition học dùng state IMU mới sau timestamp ACK và gần chu kỳ Ts; vẫn có
  độ trễ PID/servo và đo lường chưa hiệu chỉnh. Final model chỉ là warm start
  trọng số, không chứa RNG và rollout buffer để tiếp tục chính xác.
- Giữ nguyên clock/NDOF; chênh lệch thang thời gian khoảng2% đã quan sát trước
  chưa được hiệu chỉnh. Các timeout firmware tính theo STM32 ms.

## 1. Cập nhật và kiểm tra firmware trước

Repo STM32_IMU, nhánh `sensor-config-diagnostics`. Clean/Build trong CubeIDE,
đảm bảo Project Location đúng clone. Hai source mới `command_guard.c` và
`actuator_gate.c` phải xuất hiện trong build; không dùng lại Debug ELF/subdir.mk
cũ được commit từ trước. Không sửa clock, pin hoặc .ioc để dùng tính năng này.

Firmware mặc định disarmed, bỏ đường điều khiển `CMD,speed,angle` cũ. Vì vậy
phần mềm laas cũ gửi CMD sẽ không làm xe chạy với firmware này. TEL1/TEL2 không
đổi layout; Pi telemetry_rx vẫn dùng riêng được.

Giao thức có CRC16/CCITT-FALSE trên payload trước dấu phẩy CRC:

```text
CTL,1,seq,ARM,0.000000,75,CRC\r\n
CTL,1,seq,RUN,speed_mps,servo_deg,CRC\r\n
CTL,1,seq,STOP,0.000000,75,CRC\r\n
ACK,1,echo_seq,armed,reason,stm32_ms,last_accepted_seq,CRC\r\n
```

ARM chỉ ở trạng thái disarmed và neutral. RUN yêu cầu armed, seq mới, vận tốc
0..0.5m/s, servo45..105°. STOP hợp lệ được ưu tiên dù sequence cũ. Timeout300ms
hoặc lỗi cảm biến làm disarm; không tự rearm. PID có stop_generation để reset
ngay cả khi STOP→ARM→RUN xảy ra giữa hai lần đọc encoder. Mã ACK:0 accepted,
1 parse/CRC,2 bounds,3 disarmed,4 sequence; stop reason5 timeout,6 sensor.
Lệnh sai không kéo dài watchdog. Dòng quá dài/tràn ring không được xử lý suffix.

Motor task kiểm tra lại timeout, encoder/IMU age≤100ms, fusion5/0 trước PWM.
STOP/zero/disarm chặn PWM độc lập với integral PID. Đây là **watchdog phần mềm**:
không đảm bảo dừng nếu MCU bị treo/halt bởi debugger hoặc nguồn/driver lỗi;
PWM0 không phải xác nhận xe đã phanh đứng yên. Tắt nguồn motor khi nạp/debug.

## 2. Build Pi

Trong repo nmpc_rl, nhánh `experimental-deployment`:

```sh
cmake -S . -B build -DBUILD_FIXED_NMPC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

Cần native CasADi3.7.2 với SQP/QRQP trên Pi, không chỉ bộ nhận telemetry. Nếu
configure không tìm thấy SDK, chỉ rõ `-Dcasadi_DIR=/path/to/casadi/cmake`.
`BUILD_FIXED_NMPC=OFF` vẫn build telemetry và command/calibration tests, nhưng
không build hardware_runner. Không cần Python để chạy binary; Python/NumPy dùng
trên host để export model, chuẩn bị input và chạy kiểm thử.

## 3. Chuẩn bị cùng một đầu vào cho bốn mode

Chọn checkpoint/config bạn muốn đánh giá, không lấy checkpoint cuối mode trước
làm đầu vào mode tiếp theo. Công cụ không train và không chỉnh dữ liệu đường.

```sh
python3 deployment_cpp14/tools/prepare_hardware_trial.py \
  --checkpoint /path/to/checkpoint.npz \
  --reference /path/to/reference.csv \
  --hardware deployment_cpp14/examples/hardware_nominal.json \
  --start initial_heads --horizon 20 \
  --output hardware_trial_inputs
```

Có thể thêm `--config /path/to/evaluation_config.json` để giữ đúng LR/bounds/reward
đã chọn. Không có config thì dùng metadata checkpoint. Xem lại
`initial.model.json`, `hardware_source.json`, `hardware.settings`, `manifest.json`.
Task Qe/Re lấy từ cấu hình checkpoint, tách khỏi Q/R mà Actor sinh. `initial_heads`
giữ hidden layers nhưng reset output heads và P như exporter hiện có; `trained`
giữ thông số đã học. Hai cách khởi tạo có thể cho kết quả khác nhau.

Nếu reference nhanh hơn giới hạn, tool từ chối; không âm thầm giảm tốc hoặc sửa
CSV. Chọn đường phù hợp. `feature_length` dùng chiều dài đường, cùng lựa chọn
route scaling của runner đối chiếu trước. Giới hạn, LR và dynamics cần được lưu
và giữ nguyên giữa các mode. `hardware_nominal.json` là khởi điểm để kiểm tra,
không phải kết quả hiệu chỉnh hoàn tất.

## 4. Chạy shadow trước (motor tắt)

Chỉ một process sở hữu serial: dừng telemetry_rx/terminal khác. Runner dùng
flock và TIOCEXCL; không chạy đồng thời hai bộ đọc. x0/y0/yaw0 phải được thay
bằng pose thật tại mốc bản đồ của bạn, không mặc định coi xe luôn ở gốc.

```sh
./build/hardware_runner \
  --model hardware_trial_inputs/initial.model \
  --settings hardware_trial_inputs/hardware.settings \
  --reference hardware_trial_inputs/reference.csv \
  --device /dev/ttyACM0 --mode fixed --duration 30 \
  --x0 0 --y0 0 --yaw0-deg 0 \
  --output shadow_fixed_01
```

Nếu chỉ đứng yên, solver vẫn có thể đề xuất chuyển động: `tx.log` phải rỗng.
Thử xoay/đẩy xe bằng tay, kiểm tra state và lệnh có hợp lý. Không tự khẳng định
CTE từ odometry là ground-truth CTE. Gói TEL2 phải có flags/freshness hợp lệ và
SYS_STATUS5/SYS_ERR0. Không có dữ liệu khỏe120ms thì exit2, không ngầm chạy lại.

Phát lại `.log` gốc ở chế độ shadow:

```sh
./build/hardware_runner \
  --model hardware_trial_inputs/initial.model \
  --settings hardware_trial_inputs/hardware.settings \
  --reference hardware_trial_inputs/reference.csv \
  --input /path/to/raw_tel2.log --mode full_adapt \
  --output replay_shadow_01
```

Shadow mọi mode đều propose/cancel, không observe/update vì lệnh chưa áp dụng.
Replay chưa có mô hình timing USB/actuator thật, không dùng để tuyên bố hiệu quả
học online trên xe. Đầu vào phải là TEL gốc; CSV không phải TEL.

## 5. Bench command/watchdog, rồi fixed chạy chậm

Trước khi đặt xe xuống sàn: kiểm tra dấu encoder, gyro, servo; kiểm tra hướng
bánh lái với motor mất nguồn; sau đó kê bánh để xác nhận PID tốc độ thấp và dừng.
Chạy drive ở khu vực kiểm soát với cách cắt nguồn motor độc lập. Ví dụ lệnh
bên dưới chỉ dùng sau các kiểm tra đó:

```sh
./build/hardware_runner \
  --model hardware_trial_inputs/initial.model \
  --settings hardware_trial_inputs/hardware.settings \
  --reference hardware_trial_inputs/reference.csv \
  --device /dev/ttyACM0 --mode fixed --duration 30 \
  --x0 0 --y0 0 --yaw0-deg 0 --drive \
  --output drive_fixed_01
```

Startup: bỏ dữ liệu cũ1000ms → STOP/ACK đồng bộ sequence → mẫu khỏe và xe đứng
→ ARM/ACK → giải NMPC → RUN/ACK → state mới sau ACK. Không ACK100ms, stale120ms,
solver>60ms, bước điều khiển lệch quá hạn, clipping, lỗi fusion/solver hoặc mất
serial đều kết thúc và cố gửi STOP. Firmware timeout là lớp độc lập khi Pi
không gửi được. SIGINT/SIGTERM cố gửi STOP; SIGKILL chỉ còn firmware watchdog.

Thử mất truyền lệnh và xác nhận **PWM thực về0** bằng debugger/đo tại driver
với bánh được kê. Không dùng kết quả host test thay cho bài này. Tiếp tục đo
thời gian solve/age trên Pi thật; đạt trên máy host chưa chứng minh đạt trên Pi.

## 6. Thực nghiệm bốn mode và AprilTag

Sau fixed đạt, thay `--mode` lần lượt bằng actor_adapt, critic_adapt, full_adapt;
mỗi lần dùng cùng initial.model/settings/reference, output mới và cùng pose
ban đầu. Giữ cùng điều kiện, chạy nhiều lần/seed và ghi cả lần lỗi. Không tự
chạy nối tiếp bốn lượt motor nếu chưa đưa xe về mốc xuất phát.

Mỗi run có:

- `rx.log`, `tx.log`: dữ liệu nhận/gửi; ACK nằm trong RX. Nếu replay raw bằng
  telemetry_rx, các dòng ACK không phải TEL sẽ được đánh dấu invalid.
- `state.csv`: timestamp nguồn, pose, acceptance và health. `host_s` cộng với
  `host_origin_steady_ns` trong summary cho mốc steady clock trên cùng máy;
  camera phải dùng/căn cùng thang thời gian, không mặc nhiên coi mọi clock giống nhau.
- `control.csv`: state vy=0, reference, features, Q/R/P, Actor mu/sigma/latent,
  lệnh NMPC và request lượng tử hóa, clipping và thời gian giải.
- `transitions.csv`: sequence command, timestamp trước/sau/ACK, reward, TD,
  error/next_error, độ lớn cập nhật. Đây là chuyển trạng thái quan sát sau ACK,
  chưa phải xác nhận vận tốc/góc cơ cấu đã đạt yêu cầu.
- `events.csv`, `summary.txt`: ACK, lỗi, kết thúc, bộ đếm cập nhật, thống kê solve.
- Snapshot initial/final model, settings và reference. Không ghi đè output cũ.

Camera–AprilTag chưa triển khai trong thay đổi này. Cần calibrate camera, quy
đổi tag về cùng điểm trên xe, căn timestamp và đánh giá độ chính xác camera.
Dùng nó làm tham chiếu độc lập nếu mục tiêu là đo độ trôi odometry. Sau đó đo
CTE/heading/speed error theo ground truth, completion, effort/smoothness lệnh,
solver success/deadline và tính lặp lại giữa bốn mode.

## Kiểm chứng phần mềm hiện có

Host fake-HAL firmware: parser/CRC/sequence/watchdog/wrap; UART chia nhỏ và
tràn; gate PWM theo sensor; PID reset kể cả rearm nhanh; IMU/encoder/TEL cũ.
C++: calibration/quantization bounds, ACK CRC, session bốn mode và CTL→C guard→ACK.
PTY: shadow không TX/không học, drive ACK/transition, lỗi fusion, timeout,
serial bị chiếm và STOP khi thoát. Các kiểm thử này không phải kiểm chứng xe.

### Kết quả kiểm tra trên host của lần triển khai này

- Build C++14 Release với CasADi:15/15 nhóm CTest qua (trong đó7 bài kiểm tra
  serial/replay/session, gồm đường cong, ACK, timeout, busy port và solver deadline).
- Build `BUILD_FIXED_NMPC=OFF`:13/13 nhóm CTest qua.
- 17 bài đối chiếu riêng Fixed NMPC, Actor/Critic, online schedule, reference và
  route controller Python/C++ qua. Không thay code train Python.
- Firmware12 nhóm fake-HAL qua với ASan/UBSan; TEL1/TEL2 qua receiver/audit, và
  C++ CTL→C guard→C++ ACK qua. Không có ARM cross build hoặc flash trong workspace.
- Review độc lập đã sửa: PID history khi rearm nhanh, STOP do port ownership thất
  bại, và góc servo lượng tử vượt giới hạn. Có regression tests tương ứng.

Các con số này là kiểm chứng phần mềm host, không phải kết quả thực nghiệm xe.
