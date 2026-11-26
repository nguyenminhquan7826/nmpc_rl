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
