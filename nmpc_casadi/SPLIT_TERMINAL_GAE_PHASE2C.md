# Phase 2C — GAE advantage for independent Qf_psi learning (evaluation-only)

## Evidence motivating the change

Preliminary diagnostic CAD results with Phase 2B:
- split two networks, freeze all Qf: mean CTE ~1.238 cm nominal.
- same noisy Qfpsi no-learning arm: ~1.238 cm nominal.
- one-step Qfpsi policy gradient: ~1.242 cm nominal.
- on FIVE independent sampling seeds (11/22/33/44/55), PG vs
  sampled-frozen mean paired CTE difference = -0.00148 cm,
  sample SD = 0.02095 cm, 3/5 seeds improve, average return delta
  = -0.924. This does NOT establish improvement.
- the initial stochastic Qf exploration sigma=0.02 reached ±5% trust
  bound only very rarely: near-zero effect of solver clipping.

This test changes ONLY how the separate Qf-head policy gradient
calculates the advantage. It does not promise improvement.

## Mechanism

New GAE arm:
    ac_nmpc_split_qf_gae

Old exact comparator arms stay:
    ac_nmpc_split_qf_sampled_frozen  (Qfpsi sampling, no learning)
    ac_nmpc_split_qf_pg              (Qfpsi sampling, one-step TD)
    ac_nmpc_split_value_frozen_qf    (Phase 2A, no sampling)

Qfpsi latent Gaussian sample, pretrained-relative hard trust
band ±5%, optimizer LR=.0001, sampling sigma=.02, trust
regularizer=.10, all Actor and Value learning algorithms/parameters
remain unchanged.

Pre-update Value Critic produces
    delta_t = r_t + gamma*(1-terminated_t)*V_{t+1} - V_t
    A_t^GAE = delta_t + gamma*lambda*(1-terminated_t)*A_{t+1}^GAE
with recursion stopping at a rollout batch boundary (16 samples)
and bootstrapping using V_{last_next} via the last delta when the
batch is not truly terminated.

The ONLY difference vs Phase 2B one-step QfPG is
lambda=0.95 rather than lambda=0.0 for the terminal-head advantage.
Both normalize A by max(batch std(A), 1.0), preserving the previous
normalization scheme. No batch centering, LR changes, gradient
normalization modifications, or other algorithmic changes were made.

**Important:** GAE may improve temporal credit assignment, but
does not necessarily reduce gradient variance. If results remain
inconsistent, diagnose sample gradient variance, action saturation
and lambda sensitivity rather than claiming a solved variance problem.

The Actor retains ORIGINAL 1-step advantage; the Value Critic
retains ORIGINAL n-step TD-Huber. These do not use GAE.

The output layer for Qfpsi alone is trainable; ALL other terminal
weights and all hidden layers remain exactly frozen. Distinct
Qf RNG and same initial checkpoint mean all three Qf stochastic
arms are paired until the first learning batch. New file
phase2c_advantage_pairing.csv verifies exact pre-update
weights and CTE against the one-step PG arm. Previous
phase2b_sampled_pairing.csv remains intact.

## Do not touch original training code or reference

Only these evaluation files differ from Phase 2B:
- split_terminal_pg.py
- adapt_nmpc.py
- test_split_terminal_gae.py (new)
- this protocol (new)

Neither original rl_nmpc.py/baseline.py/training XREF nor the
selected R3A checkpoint is changed.

## Windows PowerShell steps

From user repository root:

    cd D:\GitHub\RL_NMPC\Code\nmpc_rl_frozen_online
    git fetch origin
    Copy-Item .\nmpc_casadi\adapt_nmpc.py D:\GitHub\RL_NMPC\Code\experiment_archive\adapt_nmpc_pre_phase2c.py
    git restore --source=origin/experiment/frozen-vs-online-ac-nmpc -- nmpc_casadi/adapt_nmpc.py nmpc_casadi/split_terminal_pg.py nmpc_casadi/test_split_terminal_gae.py
    cd nmpc_casadi

Run new tests and full regression first:

    python -m unittest -v test_split_terminal_gae

    python -m unittest -v test_reward_r1 test_critic_huber_r2a test_adapt_frozen test_heldout_reference test_cad_eval_reference test_ac_ablation test_qfpsi_isolation test_sampling_seed test_qfpsi_trust_region test_split_critic_eval test_split_terminal_pg test_split_terminal_gae

Expected previous suite plus 7 new tests if all files present;
do NOT proceed if any test is FAILED.

Set inputs:

    $ck='D:\GitHub\RL_NMPC\Code\experiment_archive\SELECTED_R3A_HUBER_LR003\checkpoint_best_tracking.npz'
    $cad='D:\GitHub\RL_NMPC\Code\experiment_archive\cad_refs\xref_states_cad_test.csv'

Short 3-second NOMINAL smoke to a NEW EMPTY output folder:

    python adapt_nmpc.py --track-csv $cad --checkpoint $ck --output ..\..\experiment_archive\eval_cad_split_2c_smoke --nominal-only --seconds 3 --rollout-steps 16 --actor-lr 0.001 --critic-lr 0.001 --seed 0 --ablation-components --split-critic --split-qf-policy-gradient --split-qf-gae --qfpg-gae-lambda 0.95 --qfpg-trust-frac 0.05 --qfpg-sigma 0.02 --qfpg-lr 0.0001 --qfpg-reg 0.10

Verify GAE arm has updates > 0, terminal Qf head training
enabled, terminal Qf head changed, hidden/other five Qf
outputs unchanged, and GAE lambda logged 0.95:

    $dir = '..\..\experiment_archive\eval_cad_split_2c_smoke'

    Import-Csv "$dir\comparison.csv" |
      Where-Object { $_.controller -like "ac_nmpc_split*" } |
      Format-List controller,updates,split_qf_head_training_enabled,split_qf_head_unchanged,split_qf_head_untouched_components_unchanged,split_qf_head_training_updates,qfpg_gae_lambda,qfpg_advantage_rms

    Import-Csv "$dir\phase2b_sampled_pairing.csv" | Format-List
    Import-Csv "$dir\phase2c_advantage_pairing.csv" | Format-List

Both files MUST report matched_weight_prefix=True and
matched_cte_prefix=True. All qfpg_applied_ratio must lie
within 0.95 to 1.05 (allowing numerical roundoff).
No plain adapted single-Agent checkpoint reproduces a
split terminal learned policy. Do not deploy on vehicle.

## AFTER smoke and regression unit tests pass

Run full 150-second NOMINAL comparison:

    python adapt_nmpc.py --track-csv $cad --checkpoint $ck --output ..\..\experiment_archive\eval_cad_split_2c_nominal --nominal-only --seconds 150 --rollout-steps 16 --actor-lr 0.001 --critic-lr 0.001 --seed 0 --ablation-components --split-critic --split-qf-policy-gradient --split-qf-gae --qfpg-gae-lambda 0.95 --qfpg-trust-frac 0.05 --qfpg-sigma 0.02 --qfpg-lr 0.0001 --qfpg-reg 0.10

Then compare new GAE arm to the old one-step PG and
same-sample/no-learning arm. Only if the result is promising,
run at least five independent sampling seeds on the same
diagnostic CAD, and held-out new CAD geometry later.
For transparent reporting, use paired difference in Mean CTE,
Max CTE, return, and runtime, and include uncertainty intervals.

Heuristic pretrained-relative band ±5% is not an NMPC
stability, recursive feasibility, or hard deadline proof.
