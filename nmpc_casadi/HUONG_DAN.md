# v8: chỉ train Critic, giữ Q/R ban đầu

Mục tiêu lần này là kiểm tra Qf học từ Critic có cải thiện điều khiển khi loại bỏ ảnh hưởng học và exploration của Actor. Công thức V=-b-e^TQf e, target Critic n16, LR Critic 0.003, biên [0.5,50], reward và tiêu chí đánh giá giữ như v7.

## Chạy thử nghiệm mới

Chạy trong thư mục nmpc_casadi, mỗi lệnh riêng một dòng:

```powershell
python -m unittest test_critic_only -v
python rl_nmpc.py --mode train --train-component critic --episodes 100 --critic-lr 0.003 --critic-nsteps 16 --rollout-steps 256 --eval-every 10 --seed 0 --initial-seed 0 --output results_v8_critic_n16
python rl_nmpc.py --mode evaluate --checkpoint results_v8_critic_n16/checkpoint_best_tracking.npz --output eval_v8_critic_n16_tracking
python rl_nmpc.py --mode evaluate --checkpoint results_v8_critic_n16/checkpoint_best_return.npz --output eval_v8_critic_n16_return
```

Code tự đọc chế độ train và n từ checkpoint khi eval. Có thể dùng run_train_critic.bat và run_evaluate_critic.bat. Lần này train mới, không truyền checkpoint đã train cả Actor–Critic. Không sửa Q/R/Qf hoặc reward giữa train và eval của cùng checkpoint.

Trong chế độ critic:
- Q và R đưa vào NMPC luôn đúng Q_INIT/R_INIT; Actor không tạo nhiễu hoặc ảnh hưởng output điều khiển.
- Actor không cập nhật: cả parameters, Adam m/v và bộ đếm t đều giữ nguyên.
- Critic cập nhật 6 multiplier Qf và baseline b. Baseline b không được đưa vào cost NMPC.
- Eval và validation dùng 3 controller: baseline, fixed_qf (50*QF_INIT), critic_only.
- Best checkpoint được chọn theo critic_only; đồ thị V-G cũng được tính trên chính policy này.
- actor_td_mean/rms vẫn được ghi như chẩn đoán TD một bước; không phải tín hiệu đang cập nhật Actor. actor_loss=0, actor_updated=0.

Train xong tự xuất comparison của checkpoint cuối; đó không phải best. Cần chạy lệnh evaluate riêng như trên.

## Đối chứng và cách kết luận

So sánh critic_only với fixed_qf trong comparison.csv và ablation_deltas.csv của eval. Return delta dương là tốt hơn; delta CTE/RMSE âm là tốt hơn. Kiểm tra hoàn thành, solver và tracking_feasible cùng với mean/max CTE, heading và vx. Nếu CTE giảm nhưng return hoặc peak CTE xấu hơn, không kết luận tối ưu tổng thể.

Qf vẫn làm thay đổi hành vi NMPC trong lúc Critic học. Việc đóng băng Actor chỉ loại bỏ ảnh hưởng của Actor, không biến đây thành đánh giá value của một policy cố định, cũng không đảm bảo TD loss tối ưu được Qf cho điều khiển.

Để so sánh train cả hai mạng với train chỉ Critic trên cùng lịch khởi tạo, chạy mới đối chứng:

```powershell
python rl_nmpc.py --mode train --train-component both --episodes 100 --actor-lr 0.001 --critic-lr 0.003 --critic-nsteps 16 --rollout-steps 256 --eval-every 10 --seed 0 --initial-seed 0 --output results_v8_both_n16
python rl_nmpc.py --mode evaluate --checkpoint results_v8_both_n16/checkpoint_best_tracking.npz --output eval_v8_both_n16_tracking
```

initial-seed tách RNG khởi tạo episode khỏi exploration, nên hai run mới dùng cùng offsets theo episode. Kết quả v7 cũ dùng RNG chung sẽ có lịch offsets khác; cùng seed không có nghĩa cùng offsets. Bộ tình huống eval cố định vẫn giống nhau. Chế độ both không truyền initial-seed giữ hành vi v7 cũ và nạp được checkpoint v7.

Nếu khảo sát target n1, chỉ đổi --critic-nsteps 1 và dùng thư mục output mới. So sánh best với best, final với final; nhiều seed cần thiết để kiểm tra độ ổn định.

## File cần gửi

training_history.csv, update_history.csv, validation_history.csv, best_checkpoints.json, config.json và ZIP thư mục eval_v8_critic_n16_tracking. Kiểm tra actor_updated/actor_parameter_delta bằng 0 và Q/R không thay đổi trong episode CSV.

Checkpoint critic-only và joint có metadata phân biệt; không nạp lẫn để tiếp tục train. Save/load khôi phục optimizer và RNG, nhưng một lần chạy tiếp bắt đầu history/selection mới trong output mới.

---

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
