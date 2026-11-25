# Chọn quỹ đạo cho training và evaluation — branch `train-and-evaluation`

Hai reference CSV được commit dưới `trajectory_references/`:

| Tên | CSV | Chiều dài | Mẫu | vx_ref |
|---|---|---:|---:|---:|
| `spiral_8m` | `xref_spiral_8m.csv` | 8 m | 801 | 0.1 m/s |
| `zigzag_8m` | `xref_zigzag_8m.csv` | 8 m | 801 | 0.1 m/s |

Các cột: `index,s_m,t_ref_s,vx_ref_mps,vy_ref_mps,psi_ref_rad,r_ref_radps,X_ref_m,Y_ref_m,kappa_1_per_m`. Thứ tự state NMPC: `[vx,vy,psi,r,X,Y]`. SHA-256 các CSV được kiểm tra trong `trajectory_selection.py`.

## Cách dùng nhanh (PowerShell, trong nmpc_casadi)

```powershell
git switch train-and-evaluation
python -m pip install -r requirements.txt
python -m unittest test_named_trajectories -v
python run_named_trajectories.py --train spiral_8m --evaluate zigzag_8m --episodes 30 --steps 1250
```

Đảo hai quỹ đạo:

```powershell
python run_named_trajectories.py --train zigzag_8m --evaluate spiral_8m --episodes 30 --steps 1250
```

So sánh trong cùng quỹ đạo: đặt `--train spiral_8m --evaluate spiral_8m`.

Dùng checkpoint đã huấn luyện để đánh giá mà không train lại:

```powershell
python evaluate_trajectory_frozen.py --checkpoint results_trajectories/spiral_8m_to_zigzag_8m_seed0/training/checkpoint_best_train.npz --trajectory zigzag_8m --output eval_spiral_to_zigzag --max-steps 1500
```

Nếu checkpoint tốt nhất không có do các episode không đạt điều kiện lựa chọn, lệnh chạy tự động dùng checkpoint cuối cùng `checkpoint_epXXXX.npz`. Luôn kiểm tra `reached_end`, `solver_success_pct`, lỗi heading và CTE trước khi so sánh reward.

## Kiểu đánh giá

Mặc định `run_named_trajectories.py` dùng `evaluate_trajectory_frozen.py` để so sánh `strong_fixed_nmpc`, `untrained_actor_critic` và `trained_frozen_actor_critic`; Actor và Critic **không cập nhật** khi evaluate, checkpoint và bounds được giữ nguyên. Đây là kiểm tra tổng quát hóa phù hợp với held-out path.

Nếu cần nghiên cứu khả năng học online khi đổi quỹ đạo (KHÔNG phải đo tổng quát hóa thuần túy):

```powershell
python run_named_trajectories.py --train spiral_8m --evaluate zigzag_8m --episodes 30 --online-eval
```

Hoặc:

```powershell
python evaluate_controller_modes.py --checkpoint TRAIN_CHECKPOINT.npz --trajectory zigzag_8m --output eval_online_zigzag --save-data
```

**Chú ý**: `evaluate_controller_modes.py` hiện chủ động thay các giới hạn và giá trị khởi tạo Q/R/P, sau đó học online ở các mode adapt. Vì thế không được gọi đó là phép đo zero-shot của checkpoint đã train.

## Tài liệu gốc và phạm vi thay đổi

- `baseline.py`, `nmpc_casadi.py`, `reference_manager.py`, `run_experiment.py` không bị thay đổi.
- `train_actor_critic.py` bổ sung `--trajectory` và vẫn chấp nhận `--reference` legacy theo SHA ghim.
- `evaluate_controller_modes.py` bổ sung `--trajectory`, vẫn chấp nhận `--reference` bên ngoài.
- `evaluate_frozen_actor_critic.load_frozen_agents` giờ chấp nhận checkpoint trên track đã đăng ký nếu SHA và metadata nhất quán.
- Mỗi lần chạy ghi vào thư mục đầu ra mới, không ghi đè kết quả cũ.
- Hai đường đã qua kiểm tra cấu trúc, chiều dài và giới hạn lái **xấp xỉ hình học**. Chưa chứng minh khả thi đệ quy/ổn định Lyapunov; các thông số learned Q/R/P cần kiểm nghiệm bằng closed-loop.
- Với `Ts=0.071s`, `v_ref=0.1m/s`, đi 8 m cần khoảng 80 giây / 1127 bước. Nên để tối thiểu `--steps 1250`, `--max-steps 1500` để tránh kết thúc do thời gian trước khi đến đích.

## Tách TRAIN và EVAL

Để đánh giá transfer, cố định hyperparameter, seed và checkpoint bằng dữ liệu TRAIN; không chọn lại checkpoint theo metric trên held-out EVAL. Có thể chạy nhiều seed khác nhau trong các output riêng. Các tham số quỹ đạo, SHA, checkpoint và kết quả `comparison.csv` được lưu kèm từng experiment.
