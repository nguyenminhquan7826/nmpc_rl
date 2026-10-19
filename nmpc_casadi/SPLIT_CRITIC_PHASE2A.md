# Phase 2A: physically split Value and Terminal Weight networks

## Why the architecture test is necessary

On the 10.7038-m diagnostic AutoCAD track under four physical profiles:

| Mode | Mean CTE over four cases |
| --- | ---: |
| Frozen | ~1.238 cm |
| Original Online | ~1.852 cm |
| Online solver-input-only Qf_psi trust 0% | ~1.171 cm |
| Earlier Critic learning-gradient Qf_psi hold | ~1.036 cm |

The current Critic uses V(s,e) = -b(s) - e^T diag(Qf(s)) e.
Therefore TD-Huber updates also adjust six terminal-weight outputs.
A solver-only trust cap makes Qf_psi safe-ish in *empirical tracking*
but does not fix the Critic's large raw-Qf drift.

## Structural intervention (NO NEW Qf LEARNING YET)

The experiment clones checkpoint's seven-output Critic into TWO
**independent** parameter sets:

- Value network: existing seven-output net, same exact V(s,e) at initial
  checkpoint, trained with completely unchanged n-step TD-Huber loss.
  Its six internal quadratic coefficients are now VALUE FEATURES ONLY.
  They are **not** wired to the NMPC solver.
- Terminal-weight network: six-output net formed by copying the same hidden
  layers and precisely slicing the first six columns of the pretrained
  output layer, including Adam moments. This outputs all SIX actual Qf(s)
  for NMPC. No gradients from TD or Actor reach it. It is FROZEN in
  Phase 2A; a separate control-performance learning objective will be
  introduced after this structural unit/smoke acceptance.

Actor remains the exact pretrained Actor and uses its original one-step
TD policy-gradient update. Actor Gaussian exploration, learning rates,
rollout batch sizes, physical plant, checkpoint, CAD, reward, NMPC model
and hard constraints are unchanged. Every arm loads a fresh checkpoint.

**Crucial scientific distinction**:
Phase 2A freezes ALL SIX terminal Qf weights, not just Qf_psi.
Hence it is NOT expected to numerically match Qf_psi Trust 0%.
The purpose is to check TD-gradient isolation and characterize the
baseline for controlled Phase 2B Qf-head learning.

No ordinary adapted Agent checkpoint is produced for split mode because
a replayable artifact would need both the updated Value/Actor network
and the separate terminal-weight network. No old outputs are overwritten.

## Run on Windows PowerShell

    cd D:\GitHub\RL_NMPC\Code\nmpc_rl_frozen_online
    git status --short
    git fetch origin

Back up existing evaluator edits before replacing them:

    Copy-Item .\nmpc_casadi\adapt_nmpc.py D:\GitHub\RL_NMPC\Code\experiment_archive\adapt_nmpc_pre_split.py

Fetch updated evaluator and new evaluation-only files:

    git restore --source=origin/experiment/frozen-vs-online-ac-nmpc -- nmpc_casadi/adapt_nmpc.py nmpc_casadi/split_critic_eval.py nmpc_casadi/test_split_critic_eval.py

    cd nmpc_casadi
    python -m unittest -v test_reward_r1 test_critic_huber_r2a test_adapt_frozen test_heldout_reference test_cad_eval_reference test_ac_ablation test_qfpsi_isolation test_sampling_seed test_qfpsi_trust_region test_split_critic_eval

Expected total: previous 51 tests plus six new tests = 57.
Tests must actually PASS on Windows before running an NMPC rollout.

Inputs:

    $ck  = 'D:\GitHub\RL_NMPC\Code\experiment_archive\SELECTED_R3A_HUBER_LR003\checkpoint_best_tracking.npz'
    $cad = 'D:\GitHub\RL_NMPC\Code\experiment_archive\cad_refs\xref_states_cad_test.csv'
    Test-Path $ck
    Test-Path $cad

Run one nominal 3-second smoke with a NEW EMPTY output directory:

    python adapt_nmpc.py --track-csv $cad --checkpoint $ck --output ..\..\experiment_archive\eval_cad_split_2a_smoke --nominal-only --seconds 3 --rollout-steps 16 --actor-lr 0.001 --critic-lr 0.001 --seed 0 --ablation-components --split-critic

Validate six arms, including ac_nmpc_split_value_frozen_qf:

    $dir = '..\..\experiment_archive\eval_cad_split_2a_smoke'
    Import-Csv "$dir\comparison.csv" | Select-Object controller,updates,split_value_terminal_networks,split_qf_head_unchanged,critic_unchanged_from_checkpoint,adapted_checkpoint_saved | Format-Table -AutoSize

    Import-Csv "$dir\ablation_sampled_pairing.csv" | Select-Object controller,matched_weight_prefix,matched_cte_prefix | Format-Table -AutoSize

For the split arm require:
- updates > 0 (3 for a 3-second smoke with rollout_steps 16);
- split_value_terminal_networks = True;
- split_qf_head_unchanged = True;
- critic_unchanged_from_checkpoint = False (independent value learns);
- adapted_checkpoint_saved = False;
- matched_weight_prefix AND matched_cte_prefix = True.

If the smoke passes, run a NOMINAL full track (150 seconds max) in a
new output directory. Do not yet run multiple plant cases:

    python adapt_nmpc.py --track-csv $cad --checkpoint $ck --output ..\..\experiment_archive\eval_cad_split_2a_nominal --nominal-only --seconds 150 --rollout-steps 16 --actor-lr 0.001 --critic-lr 0.001 --seed 0 --ablation-components --split-critic

Compare Frozen/Original Online/Split baseline on CTE, heading, reward,
completion, solver success and deadline metrics.

## Next: Phase 2B — separate learning of terminal Qf

After Phase 2A passes, the terminal network may be optimized via
stochastic policy gradient using a control-performance return signal
(with the independent Value network as a baseline). Constrain
candidate Qf proposals relative to original pretrained Qf(s), use
appropriate sampling density (or score-function estimator), include
trust penalty and safety checks, and use the fallback when screening
fails. The optimization objective and proof obligations are
research extensions, NOT equations or guarantees supplied by the paper.

The old diagnostic CAD was used to develop the method. Final claims
need fresh unseen CAD geometry and independent RNG sampling seeds.
