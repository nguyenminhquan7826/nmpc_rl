# Phase 2D — signed Qf_psi sensitivity, before changing the learning objective

## Decision from previous stages

Diagnostic external CAD nominal mean CTE:
- Frozen AC–NMPC: 1.2434 cm.
- Original online AC–NMPC: 1.8651 cm.
- Independent Value + frozen terminal Qf (Phase 2A): 1.2379 cm.
- Same independent Value + stochastic Qfpsi, no Qf learning (2B): 1.2378 cm.
- Same stochastic Qfpsi and one-step PG: 1.2421 cm.
- Same stochastic Qfpsi and GAE lambda=0.95: 1.2437 cm.

On independent Actor-sampling seeds 11,22,33,44,55, one-step PG
mean paired CTE difference relative to the same-sampled-no-learning
arm was -0.00148 cm with sample SD ~0.02095 cm, no consistent
improvement. Do not change PG loss, learning rate or train all six Qf
without first observing the signed sensitivity of actual NMPC control.

## Scientific question

Keeping Actor and Value Critic updates the SAME as split Phase 2A,
and all other five pretrained Qf entries unchanged, how does tracking
respond to signed perturbations of the terminal heading weight?

For each current state observation s:

    Qf_psi(s) = (1 + epsilon) * Qf_psi_pretrained(s)

    epsilon ∈ {-0.05, -0.02, 0.00, +0.02, +0.05}

The pretrained Qf_psi is a STATE-DEPENDENT FUNCTION of s, NOT a
single fixed numeric constant. The value network continues unchanged
TD-Huber adaptation, Actor continues its one-step PG. The independent
terminal Qf network is frozen. No Qf stochastic sampling, no Qf
learning, no GAE and no second clip is applied during this sweep.

Each signed arm loads the exact same pretrained checkpoint and Actor
RNG seed, and begins from the same reference and plant condition.
After the first step, different Qf offsets naturally produce different
states and hence different actions. They are paired by checkpoint,
seed, and plant, not identical trajectories.

The 0% arm is a strict replicate of Phase 2A. It MUST have exact
initial Q/R/Qf and CTE pairing with original Frozen and preferably
match Phase 2A for the entire rollout. The nonzero arms intentionally
do NOT have the identical weight/CTE pre-update prefix with Frozen.

## Code scope

Only evaluation files:
- signed_qfpsi_sweep.py (new)
- test_signed_qfpsi_sweep.py (new)
- adapt_nmpc.py (updated)
- SIGNED_QFPSI_PHASE2D.md (new)

No changes to rl_nmpc.py, baseline.py, training reference,
original R3A checkpoint, or other Phase 2B/2C control modes.

Signed arm labels:
- ac_nmpc_split_qfpsi_m05  (−5%)
- ac_nmpc_split_qfpsi_m02  (−2%)
- ac_nmpc_split_qfpsi_p00  (0%)
- ac_nmpc_split_qfpsi_p02  (+2%)
- ac_nmpc_split_qfpsi_p05  (+5%)

## Windows instructions

Open Windows PowerShell in original repository:

    cd D:\GitHub\RL_NMPC\Code\nmpc_rl_frozen_online
    git fetch origin

Back up current local evaluator, if it has any local work:

    Copy-Item .\nmpc_casadi\adapt_nmpc.py D:\GitHub\RL_NMPC\Code\experiment_archive\adapt_nmpc_pre_phase2d.py

Check out the evaluation-only files:

    git restore --source=origin/experiment/frozen-vs-online-ac-nmpc -- nmpc_casadi/adapt_nmpc.py nmpc_casadi/signed_qfpsi_sweep.py nmpc_casadi/test_signed_qfpsi_sweep.py
    cd nmpc_casadi

First test structural invariants:

    python -m unittest -v test_signed_qfpsi_sweep

Then run regression including old test modules:

    python -m unittest -v test_reward_r1 test_critic_huber_r2a test_adapt_frozen test_heldout_reference test_cad_eval_reference test_ac_ablation test_qfpsi_isolation test_sampling_seed test_qfpsi_trust_region test_split_critic_eval test_split_terminal_pg test_split_terminal_gae test_signed_qfpsi_sweep

Do not run simulator if any unit test fails. It is acceptable to
report a test failure to resolve it; do not silently weaken checks.

Inputs:

    $ck  = "D:\GitHub\RL_NMPC\Code\experiment_archive\SELECTED_R3A_HUBER_LR003\checkpoint_best_tracking.npz"
    $cad = "D:\GitHub\RL_NMPC\Code\experiment_archive\cad_refs\xref_states_cad_test.csv"

NEW output folder smoke test (3 simulated seconds, nominal only):

    python adapt_nmpc.py --track-csv $cad --checkpoint $ck --output ..\..\experiment_archive\eval_cad_split_2d_smoke --nominal-only --seconds 3 --rollout-steps 16 --actor-lr 0.001 --critic-lr 0.001 --seed 0 --ablation-components --split-critic --qfpsi-signed-sweep

11 controllers are selected, including original comparators and the
five signed-Qf arms. All use same R3A checkpoint. Only 5 are tested
for signed control sensitivity.

Audit checks:

    $dir="..\..\experiment_archive\eval_cad_split_2d_smoke"
    Import-Csv "$dir\signed_qfpsi_sensitivity.csv" | Format-Table scenario,signed_offset_pct,mean_cte_post_cm,delta_mean_cte_post_cm_vs_zero,updates -AutoSize
    Import-Csv "$dir\ablation_sampled_pairing.csv" | Where-Object controller -eq "ac_nmpc_split_qfpsi_p00" | Format-List
    Import-Csv "$dir\nominal_ac_nmpc_split_qfpsi_m05_signed_audit.csv" | Select-Object -Last 3 qfsweep_pretrained_psi,qfsweep_applied_psi,qfsweep_ratio | Format-Table -AutoSize

Expected:
- sampled prefix of 0% arm matches original Frozen exactly;
- all five signed arms have 3 updates for smoke, nonfrozen Value
  and Actor parameters updated while terminal head stays frozen;
- signed audit qfsweep_ratio equals 0.95,0.98,1.00,1.02,1.05
  respectively for EVERY applied NMPC step;
- no plain single-Agent checkpoint for split controllers.

After full regression tests AND short smoke succeed,
run nominal 150-second full CAD:

    python adapt_nmpc.py --track-csv $cad --checkpoint $ck --output ..\..\experiment_archive\eval_cad_split_2d_nominal --nominal-only --seconds 150 --rollout-steps 16 --actor-lr 0.001 --critic-lr 0.001 --seed 0 --ablation-components --split-critic --qfpsi-signed-sweep

Inspect:

    Import-Csv "..\..\experiment_archive\eval_cad_split_2d_nominal\signed_qfpsi_sensitivity.csv" | Format-Table signed_offset_pct,mean_cte_post_cm,max_cte_post_cm,heading_rmse_post_rad,reward,delta_mean_cte_post_cm_vs_zero,reached_end -AutoSize

Interpret signed comparisons:
- negative CTE delta: candidate improves over 0%;
- positive CTE delta: worse than 0%;
- examine MaxCTE, heading RMSE, reward, solver success and time;
- if negative offsets consistently improve CTE, negative tuning
  of terminal heading penalty merits further exploration;
- if no clear monotone relation, a state/curvature-dependent Qf
  policy is more justified than a global upward/downward adjustment.

These are in-simulation sensitivity runs on one diagnostic CAD and
one seed. Confirm on multiple independent sampling seeds and new
held-out AutoCAD geometries before making general claims. A bound
around pretrained Qf is NOT a closed-loop stability certificate.
