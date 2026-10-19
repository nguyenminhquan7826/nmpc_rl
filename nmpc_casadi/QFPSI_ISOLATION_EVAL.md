# Causal diagnostic: hold only the pretrained Qf_psi(s) FUNCTION

This is a continuation of the Actor-only/Critic-only ablation using the
same held-out (now diagnostic) AutoCAD track and the SAME R3A Huber checkpoint.

Observed in the 24-rollout original ablation:
- Frozen mean CTE across four plant profiles: 1.238 cm
- Actor-only mean CTE: 1.225 cm
- Critic-only mean CTE: 1.848 cm
- Both-Online mean CTE: 1.853 cm
- Mean Qf_psi after 1.3 m: Frozen 1948; Critic-only 4047; Both 4096

This shows the Critic UPDATE is implicated, but NOT which particular Qf
component caused the change. Qf_X also grew 6–8% with Critic-only.

**Intervention**: compare each active-Critic mode with and without
holding Qf_psi(s) to the original checkpoint function evaluated at the
CURRENT observed state. Other Qf heads and the Critic value baseline still
learn; the Actor still learns in both-Online.

Qf_psi has shared hidden layers. Simply freezing the last column in the
Critic network does NOT hold the state-dependent function when the hidden
layers change. The evaluator deep-copies the pretrained Critic, uses its
original output 2 for every controller action and value computation, and
blocks gradient for output 2 in Agent.learn(). It audits each applied action
against the pretrained function at that same observation.

The two masked modes are **diagnostic-only model compositions**. They do NOT
save a misleading ordinary adapted checkpoint, because replay would need
both the updated Critic and a copy of the original Qf_psi(s) function.
The original checkpoint remains unchanged; no training script or source
XREF is modified. Masked-mode wall-time includes the extra frozen network
forward pass; it is NOT an optimized deployment execution-time estimate.

## PowerShell

Stay in D:\GitHub\RL_NMPC\Code\nmpc_rl_frozen_online

    git status --short
    git fetch origin

Back up a modified adapt_nmpc.py before replacing it, e.g.:

    Copy-Item .\nmpc_casadi\adapt_nmpc.py D:\GitHub\RL_NMPC\Code\experiment_archive\adapt_nmpc_pre_qfpsi_hold.py

Fetch only the updated evaluator and the two new Qf-psi isolation files:

    git restore --source=origin/experiment/frozen-vs-online-ac-nmpc -- nmpc_casadi/adapt_nmpc.py nmpc_casadi/qfpsi_isolation.py nmpc_casadi/test_qfpsi_isolation.py
    cd nmpc_casadi

Validate current and new tests:

    python -m unittest -v test_reward_r1 test_critic_huber_r2a test_adapt_frozen test_heldout_reference test_cad_eval_reference test_ac_ablation test_qfpsi_isolation

Reuse the original diagnostic checkpoint and CAD:

    $ck = 'D:\GitHub\RL_NMPC\Code\experiment_archive\SELECTED_R3A_HUBER_LR003\checkpoint_best_tracking.npz'
    $cad = 'D:\GitHub\RL_NMPC\Code\experiment_archive\cad_refs\xref_states_cad_test.csv'
    Test-Path $ck
    Test-Path $cad

Run a SMALL isolated new smoke output; do not reuse previous run folders:

    python adapt_nmpc.py --track-csv $cad --checkpoint $ck --output ..\..\experiment_archive\eval_cad_qfpsi_smoke --nominal-only --seconds 3 --rollout-steps 16 --actor-lr 0.001 --critic-lr 0.001 --seed 0 --ablation-components --isolate-qfpsi

Verify that 7 rows appear (fixed, frozen sampled, actor-only,
critic-only, critic_qfpsi_held, both, both_qfpsi_held).
The pairwise initial 16-step sampled prefixes all must match Frozen.
The held modes must have qfpsi_function_held_at_checkpoint=True
and adapted_checkpoint_saved=False. No ordinary adapted checkpoint
is created for either held arm.

    $dir = '..\..\experiment_archive\eval_cad_qfpsi_smoke'
    Import-Csv "$dir\comparison.csv" | Select-Object controller,updates,qfpsi_function_held_at_checkpoint,adapted_checkpoint_saved | Format-Table -AutoSize
    Import-Csv "$dir\ablation_sampled_pairing.csv" | Select-Object controller,matched_weight_prefix,matched_cte_prefix | Format-Table -AutoSize

If this PASSES, run 7 controllers x 4 plant cases = 28 rollouts
(or add --include-frozen-deterministic for 8 x 4 = 32).
Set a NEW empty output folder:

    python adapt_nmpc.py --track-csv $cad --checkpoint $ck --output ..\..\experiment_archive\eval_cad_qfpsi_isolation_r3a --scenario-set physical --seconds 150 --rollout-steps 16 --actor-lr 0.001 --critic-lr 0.001 --seed 0 --ablation-components --isolate-qfpsi

Look at:
- comparison.csv for Mean/Max CTE, heading error, reward and completion.
- ablation_qfpsi_summary.csv for applied Qf_psi(s), Q_psi, R_delta.
- ablation_sampled_pairing.csv for matched first 16 steps.
- pairwise_learning_effect.csv for critic_qfpsi_held vs critic_only and
  both_qfpsi_held vs both.
- *_updates.csv for learning dynamics of still-adapting heads.

## Pre-registered interpretation

- If both held-Qfpsi modes substantially recover Frozen-like CTE, Qf_psi
  growth is a plausible major cause on this diagnostic track.
- If Critic-only with Qfpsi held STILL produces a high CTE, other Critic
  weights, value changes, and network interactions are involved.
- If held Qfpsi improves heading-vs-CTE tradeoff but degrades reward/solver,
  evaluate all metrics, not only CTE.
- The held output is state-dependent, so its time average can differ from
  Frozen if vehicle states differ. The audited equality is at a common STATE,
  not necessarily equal time-averaged Qfpsi for different trajectories.

This diagnostic CAD has informed algorithm choices. Any positive result
requires validation on additional independent CAD paths and multiple seeds
before claiming cross-track generalization.
