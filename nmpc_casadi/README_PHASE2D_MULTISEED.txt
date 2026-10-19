PHASE 2D — KIỂM TRA 5 SEED x 4 ĐIỀU KIỆN PLANT
================================================

Gói gồm:
  phase2d_physical_multiseed.py       Script chạy + xác minh + tổng hợp
  test_phase2d_physical_multiseed.py  14 kiểm thử với dữ liệu giả lập

PHẠM VI
- Không sửa train code rl_nmpc.py, NMPC core, checkpoint, CAD gốc.
- Dùng chính evaluator adapt_nmpc.py từ Phase 2D.
- Các mức Qf_psi: -5%, -2%, 0%, +2%, +5%.
- Plant: nominal, payload_ramp, tire_segment, rolling_segment.
- Seed ngẫu nhiên (Actor Gaussian): 11, 22, 33, 44, 55.
- Đối chứng Qf_psi 0% phải giống hệt Split Phase 2A trên toàn quỹ đạo.
- Không có mạng Qf terminal nào được học ở thí nghiệm này.

CÀI ĐẶT
1. Giải nén hai file .py vào:
   D:\GitHub\RL_NMPC\Code\nmpc_rl_frozen_online\nmpc_casadi
   (cùng thư mục với adapt_nmpc.py). KHÔNG ghi đè adapt_nmpc.py.

2. Trong PowerShell:
   cd D:\GitHub\RL_NMPC\Code\nmpc_rl_frozen_online\nmpc_casadi

   $ck  = 'D:\GitHub\RL_NMPC\Code\experiment_archive\SELECTED_R3A_HUBER_LR003\checkpoint_best_tracking.npz'
   $cad = 'D:\GitHub\RL_NMPC\Code\experiment_archive\cad_refs\xref_states_cad_test.csv'

3. Kiểm thử script (không cần solver):
   python -m unittest -v test_phase2d_physical_multiseed

4. Kiểm tra lệnh mô phỏng mà không chạy và không ghi file:
   python phase2d_physical_multiseed.py --checkpoint $ck --track-csv $cad --dry-run

5. CHẠY THẬT (20 lượt điều kiện plant x seed, mỗi lượt thử 5 offset,
   evaluator tự chạy các baseline cần thiết, nên mất thời gian):
   python phase2d_physical_multiseed.py --checkpoint $ck --track-csv $cad

Thư mục mới mặc định:
   ..\..\experiment_archive\eval_cad_split_2d_physical_multiseed\
       seed_11\
       seed_22\
       seed_33\
       seed_44\
       seed_55\
       phase2d_summary_001\

CHỐNG GHI ĐÈ
Nếu đã có thư mục seed_*, script sẽ từ chối chạy mặc định.
- Chỉ tổng hợp từ kết quả có sẵn và tạo thư mục summary MỚI:
   python phase2d_physical_multiseed.py --checkpoint $ck --track-csv $cad --summary-only
- Tiếp tục sau khi gián đoạn, chỉ bỏ qua những seed đã HOÀN TẤT
  và kiểm chứng toàn bộ dữ liệu hợp lệ:
   python phase2d_physical_multiseed.py --checkpoint $ck --track-csv $cad --resume
- Nếu seed trước đã tạo ra thư mục NHƯNG chưa hoàn tất,
  KHÔNG dùng --resume để ghi đè; hãy đổi sang một --output-root mới.
- Đổi thư mục output nếu cần:
   python phase2d_physical_multiseed.py --checkpoint $ck --track-csv $cad --output-root '..\..\experiment_archive\eval_cad_split_2d_physical_multiseed_v2'

BÁO CÁO SINH RA
phase2d_summary_001\phase2d_seed_scenario_offset.csv
  - 100 dòng: 5 seed x 4 scenario x 5 offset. Có delta CTE,
    MaxCTE, heading, reward, solver success, deadline.
phase2d_summary_001\phase2d_scenario_offset_summary.csv
  - 20 dòng: trung bình trên 5 seed CHO MỖI scenario và offset,
    kèm số seed cải thiện, độ lệch chuẩn và CI95 khi n=5.
phase2d_summary_001\phase2d_macro_by_offset.csv
  - 5 dòng: trung bình 4 scenario trong từng seed TRƯỚC,
    sau đó mới trung bình 5 seed; tránh giả độc lập n=20.
phase2d_summary_001\phase2d_provenance.json
  - SHA256 checkpoint, CAD, evaluator, seed và siêu tham số.

Đọc kết quả chính:
  $f = '..\..\experiment_archive\eval_cad_split_2d_physical_multiseed\phase2d_summary_001\phase2d_macro_by_offset.csv'
  Import-Csv $f | Select-Object offset_pct,average_delta_mean_cte_cm,average_delta_max_cte_cm,average_delta_heading_rmse_rad,average_delta_reward,mean_cte_improved_seed_count,max_cte_improved_seed_count,all_completed,min_solver_success_pct,worst_deadline_miss_pct | Format-Table -AutoSize

LƯU Ý KHOA HỌC
- 5 seed được xét trên cùng một hình học CAD (chưa kiểm tra CAD thứ hai).
- 4 plant scenario không tương đương 4 quỹ đạo độc lập.
- Cải thiện CTE trung bình không chứng minh ổn định hoặc tối ưu.
- Deadline 71 ms có thể bị vượt dù solver success = 100%.
- Script là công cụ kiểm chứng; chưa triển khai Phase 2E.
