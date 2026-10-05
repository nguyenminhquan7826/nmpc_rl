# v7: khảo sát target nhiều bước cho Critic

Giải nén toàn bộ vào cùng thư mục. Bản này giữ baseline V=-b-e^TQf e, cấu trúc mạng, trọng số INIT/reward, LR mặc định, biên, các phép ablation và lựa chọn checkpoint theo validation. Thay đổi huấn luyện chính: target Critic dùng n bước; Actor vẫn dùng TD một bước.

## Chạy thử nghiệm chính — train mới

```powershell
python rl_nmpc.py --mode train --episodes 100 --actor-lr 0.001 --critic-lr 0.003 --rollout-steps 256 --critic-nsteps 16 --eval-every 10 --seed 0 --output results_v7_n16
```

Sau khi train xong, chạy riêng từng lệnh:

```powershell
python rl_nmpc.py --mode evaluate --checkpoint results_v7_n16/checkpoint_best_tracking.npz --critic-nsteps 16 --output eval_v7_n16_tracking
python rl_nmpc.py --mode evaluate --checkpoint results_v7_n16/checkpoint_best_return.npz --critic-nsteps 16 --output eval_v7_n16_return
```

Nạp đúng checkpoint đã train với n tương ứng. Đây là checkpoint version 7; không dùng checkpoint v6 để tiếp tục train lần so sánh này. Cần train từ cùng khởi tạo để so sánh target một bước và 16 bước. Bản evaluate mặc định n=16; với checkpoint n=1 cần truyền --critic-nsteps 1.

Nếu cần đối chứng một bước cùng phiên bản code:

```powershell
python rl_nmpc.py --mode train --episodes 100 --actor-lr 0.001 --critic-lr 0.003 --rollout-steps 256 --critic-nsteps 1 --eval-every 10 --seed 0 --output results_v7_n1
python rl_nmpc.py --mode evaluate --checkpoint results_v7_n1/checkpoint_best_tracking.npz --critic-nsteps 1 --output eval_v7_n1_tracking
```

So sánh cùng cách chọn checkpoint: best_tracking với best_tracking, hoặc final với final. Không so best của n16 với final của n1 rồi quy hết cải thiện cho target. Một seed là thử nghiệm ban đầu; muốn kết luận ổn định cần lặp cùng các seed cho hai cấu hình.

## n bước khác rollout và horizon NMPC

- NMPC N=10: số bước dự đoán trong bài toán điều khiển.
- rollout-steps=256: thu tối đa 256 transition rồi cập nhật một lần.
- critic-nsteps=16: mỗi target Critic tích lũy tối đa 16 reward, tương ứng 0.32 giây mô phỏng tại Ts=0.02.

Với m bước có sẵn, target Critic:

    y_k = sum_{j=0}^{m-1} gamma^j r_{k+j} + gamma^m V(s_{k+m})

Nếu gặp termination thật thì bỏ bootstrap. Nếu chỉ hết batch hoặc hết thời gian mô phỏng thì bootstrap tại state cuối đang có. m<=16; các mẫu cuối batch dùng ít hơn 16 reward. Không ghép qua batch đã cập nhật mạng và không ghép episode. Target tính bằng giá trị mạng trước update, không backprop qua target. Critic loss=0.5*mean((y-V)^2).

Actor vẫn dùng advantage từ reward + gamma*(1-terminated)*V(next) - V(current), cũng tính trước update. Chỉ công thức target Critic được thay đổi trực tiếp; về sau value học khác vẫn ảnh hưởng tín hiệu Actor. n=16 là mức thử đầu tiên, chưa khẳng định tối ưu.

## Log mới và cách đọc

- td_rms/td_mean/critic_loss trong TRAIN: residual của target Critic n bước.
- actor_td_rms/actor_td_mean: residual một bước dùng cho Actor.
- critic_target_horizon_mean/min/max: số reward thực tế trong target.
- critic_full_nstep_pct: tỷ lệ mẫu dùng đủ n reward.
- Đồ thị nominal_critic_value_vs_return.png và critic_eval_td_rms trong EVAL vẫn là TD một bước để so sánh các phiên bản. Độ khớp V-G dùng return chiết khấu đến kết thúc thật hoặc bootstrap tail nếu truncated.

TD RMS n16 và TD RMS n1 không cùng target; không dùng độ lớn của hai chỉ số này riêng lẻ để kết luận mạng nào tốt. Kiểm tra value_rmse/bias trong eval và chất lượng điều khiển trên cùng tình huống.

## Ablation / checkpoint

Eval xuất 5 controller ×8 tình huống: baseline, fixed_qf, actor_fixed_qf, critic_only, actor_critic. actor_fixed_qf là đối chứng cần giữ.

Train lưu checkpoint.npz (final), checkpoint_best_tracking.npz, checkpoint_best_return.npz, best_checkpoints.json. Quy tắc chọn: ưu tiên tracking_feasible, rồi hoàn thành >=99% với solver>=99%, rồi CTE thấp nhất hoặc return cao nhất. Best chỉ xét các episode đã validation nominal, không phải mọi episode. Best_return là lựa chọn có ưu tiên feasibility, không nhất thiết raw return cao nhất.

## Gửi để đánh giá

training_history.csv, update_history.csv, validation_history.csv, best_checkpoints.json và ZIP thư mục eval_v7_n16_tracking. Nếu chạy đối chứng n1, gửi tương ứng cả hai cấu hình. Các file config.json ghi n và checkpoint_source để tránh nhầm.

## Kiểm tra đã chạy

Cú pháp; target tính tay với termination, batch/time truncation; n=1 cập nhật chính xác như v6; Actor có cùng gradient khi bắt đầu từ cùng mạng với n=1 hoặc n=16; gradient Critic target n16 so với sai phân hữu hạn (sai số ~7.2e-11); save/load và từ chối checkpoint sai n; train 2 episode giới hạn 0.4s với batch 8; train bổ sung 1 episode với batch 256 có đủ target 16 bước; eval 8 tình huống ×5 controller giới hạn 0.4s. Các kiểm tra ngắn không đánh giá toàn quỹ đạo hay huấn luyện 100 episode.
