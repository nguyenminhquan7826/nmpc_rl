# Held-out Fixed / Frozen / Online evaluation

## Study design

Old experiments changed plant parameters but **reused the training CAD route**. They measured robustness on a known reference, NOT generalization to unseen geometry.

The revised evaluator defaults to **--track heldout_s_bend** and generates an analytically independent open S-bend, not a transform or resampling of the training CAD:
  
    kappa(s) = 1.15 sin(4 pi s / L) - 0.45 sin(2 pi s / L)

with L equal to the approximately 7.98 m training path length, reference speed 0.10 m/s, both left and right bends, and bounded curvature/steering. The original training CSV, solver and pretrained Huber version-11 checkpoint remain unchanged. To reproduce the legacy path explicitly, pass **--track training**.

All Fixed, sampled Frozen and Online controllers use the **same new path**, checkpoint (where appropriate), initial offsets, plant and nominal predictor. Frozen is sampled to match Online exploration but has no learning. Optionally add **--include-frozen-deterministic** to evaluate frozen mean-policy inference without Gaussian sampling.

The evaluator writes new source and tail CSVs, plus **reference_manifest.json** with hashes, curvature statistics, training-reference SHA-256, and numeric evidence of path difference.

## Update the EXISTING worktree, do not create any new folders for code

From PowerShell:

    cd D:\GitHub\RL_NMPC\Code\nmpc_rl_frozen_online
    git status --short -- nmpc_casadi/adapt_nmpc.py nmpc_casadi/heldout_reference.py nmpc_casadi/test_heldout_reference.py
    git fetch origin
    git restore --source=origin/experiment/frozen-vs-online-ac-nmpc -- nmpc_casadi/adapt_nmpc.py nmpc_casadi/heldout_reference.py nmpc_casadi/test_heldout_reference.py nmpc_casadi/FROZEN_ONLINE_EVAL.md
    cd nmpc_casadi
    python -m unittest -v test_reward_r1 test_critic_huber_r2a test_adapt_frozen test_heldout_reference

Do not run git restore on locally edited files until you have backed up those edits.

Set the archived episode-79 checkpoint:

    $ck = 'D:\GitHub\RL_NMPC\Code\experiment_archive\SELECTED_R3A_HUBER_LR003\checkpoint_best_tracking.npz'
    Test-Path $ck

## Smoke on novel path (nominal plant)

    python adapt_nmpc.py --checkpoint $ck --output ..\..\experiment_archive\eval_heldout_smoke --track heldout_s_bend --nominal-only --seconds 3 --rollout-steps 16 --actor-lr 0.001 --critic-lr 0.001 --seed 0

The short test checks integration only; it cannot validate completion of the new route. Inspect comparison.csv, initial_sampled_pairing.csv, and reference_manifest.json.

## Full held-out test: four physical plant scenarios

    python adapt_nmpc.py --checkpoint $ck --output ..\..\experiment_archive\eval_heldout_s_bend --track heldout_s_bend --scenario-set physical --seconds 120 --rollout-steps 16 --actor-lr 0.001 --critic-lr 0.001 --seed 0 --include-frozen-deterministic

The deterministic mode is an optional fourth controller, not a substitute for the sampled Frozen vs Online comparison.

## Output review

- reference_manifest.json: verifies held-out geometry and provenance.
- heldout_s_bend_v1_reference.csv: entirely new geometric route.
- heldout_s_bend_v1_with_virtual_tail.csv: includes prediction-only end tail.
- comparison.csv: full metrics including tracking, completion, heading, speed, solver and steering variation.
- pairwise_learning_effect.csv: frozen-vs-fixed and online-vs-frozen contrasts.
- initial_sampled_pairing.csv: action and CTE prefix equivalence before first Online update.

A single new reference is a first OOD test, not a proof of universal generalization. Its difficulty may differ from training even when speed and length are matched. Examine completion, solver and deadline performance alongside CTE, then repeat on other unseen tracks and multiple seeds.


## Real CAD train/test split (NEW: --track-csv)

**Training remains on the ORIGINAL CAD:** no changes to baseline.py,
rl_nmpc.py, xref_states_smoothed.csv or xref_states_controller.csv.
The original R3A Huber checkpoint can be evaluated on a new 10.704 m
AutoCAD track without retraining. Do not copy the new CAD over either
training reference CSV.

PowerShell (reuse your ONLY evaluation worktree, with no new worktree):

    cd D:\GitHub\RL_NMPC\Code\nmpc_rl_frozen_online
    git status --short
    git fetch origin
    git restore --source=origin/experiment/frozen-vs-online-ac-nmpc -- nmpc_casadi/adapt_nmpc.py nmpc_casadi/cad_eval_reference.py nmpc_casadi/test_cad_eval_reference.py nmpc_casadi/FROZEN_ONLINE_EVAL.md
    cd nmpc_casadi
    python -m unittest -v test_reward_r1 test_critic_huber_r2a test_adapt_frozen test_heldout_reference test_cad_eval_reference

Review and back up any personal edits BEFORE using git restore.

Copy the uploaded xref_states_cad_test.csv to an independently named
location, e.g.:

    D:\GitHub\RL_NMPC\Code\experiment_archive\cad_refs\xref_states_cad_test.csv

Keep that file outside the training project. The evaluator copies it into
a NEW run folder and adds an evaluation-only virtual horizon tail. Original
XREF bytes and profile v_ref(s) are preserved; no speed rewrite to 0.10 m/s.
The CSV is validated against the current NMPC test speed envelope
0.05 <= vx_ref <= 0.10 m/s and steering bounds. A speed profile below
0.05 m/s requires changing the model/experiment settings intentionally,
not silently clipping or elevating the requested speed.

Run one short smoke check:

    $ck = 'D:\GitHub\RL_NMPC\Code\experiment_archive\SELECTED_R3A_HUBER_LR003\checkpoint_best_tracking.npz'
    $cad = 'D:\GitHub\RL_NMPC\Code\experiment_archive\cad_refs\xref_states_cad_test.csv'
    python adapt_nmpc.py --track-csv $cad --checkpoint $ck --output ..\..\experiment_archive\eval_cad_smoke --nominal-only --seconds 3 --rollout-steps 16 --actor-lr 0.001 --critic-lr 0.001 --seed 0

This is an integration-only test (the route is >10 m). Check
reference_manifest.json: id=cad_eval_external,
profile_preserved_from_source=true, training and reference source SHA-256.
Check initial_sampled_pairing.csv for matched_weight_prefix and matched_cte_prefix.

Run the full CAD Evaluation on four physical plant conditions:

    python adapt_nmpc.py --track-csv $cad --checkpoint $ck --output ..\..\experiment_archive\eval_cad_fixed_frozen_online --scenario-set physical --seconds 150 --rollout-steps 16 --actor-lr 0.001 --critic-lr 0.001 --seed 0 --include-frozen-deterministic

Never reuse an occupied output folder. A single new CAD track assesses
transfer to that geometry, not universal generalization; use additional
independent CADs and seeds for robust conclusions. Any speed variation
in the new evaluation source changes both geometry and reference-speed
distribution relative to pretrained 0.10 m/s training. If necessary
use a separate constant-speed evaluation control to isolate geometry.
