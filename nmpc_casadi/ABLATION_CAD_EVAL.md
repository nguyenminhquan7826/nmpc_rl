# Evaluation-only Actor/Critic ablation on a held-out AutoCAD track

## Research question

The first full CAD evaluation showed that Online (Actor+Critic updates) had
higher mean lateral tracking error than the same checkpoint held Frozen.
The terminal heading weight \`qf_2 = Qf_psi\` also increased substantially.
That correlation alone is NOT evidence of causation.

Control arms using the same pretrained R3A Huber checkpoint:

| Mode | Actor Q/R network updates | Critic Qf/value network updates | Sampling |
|---|---|---|---|
| Fixed NMPC | N/A | N/A | None |
| Frozen AC-NMPC | NO | NO | Gaussian |
| Actor-only Online | YES | NO | Gaussian |
| Critic-only Online | NO | YES | Gaussian |
| Both Online | YES | YES | Gaussian |
| Frozen deterministic (optional) | NO | NO | Actor mean |

The *inactive* network, its Adam moments and step count are checked for exact
immutability. For both-mode, the call is unchanged from the previous evaluator.
All AC arms independently reload the same pretrained checkpoint; they are
expected to have identical sampled weights/CTE until the first update (step 16).
Each plant scenario uses the same initial pose and mismatch schedule.

**Important implementation limitation:** The evaluation-only masking still
computes gradients for *both* networks inside \`Agent.learn\`, then suppresses
\`MLP.update\` on the inactive network. Thus these runs are valid to compare
**tracking behavior**, but Actor-only/Critic-only learning wall-time includes
unused calculations and does NOT measure optimized embedded WCET.

The experiment uses held-out CAD geometry. This track has now been used to
diagnose policy choices, so reserve another independent CAD and new seeds for
any final generalization claim.

## Local PowerShell setup (keep original train worktree untouched)

Reuse ONLY the existing \`nmpc_rl_frozen_online\` detached worktree.
From \`D:\GitHub\RL_NMPC\Code\nmpc_rl_frozen_online\`:

    git status --short
    git fetch origin

Back up any edits to these exact paths before restoring them. In particular,
do not overwrite a locally customized \`adapt_nmpc.py\` without saving it:

    $backup = 'D:\GitHub\RL_NMPC\Code\experiment_archive\backup_adapt_before_ablation_20261019.py'
    Copy-Item .\nmpc_casadi\adapt_nmpc.py $backup -Force

Update ONLY the evaluator and new ablation unit test/helper:

    git restore --source=origin/experiment/frozen-vs-online-ac-nmpc -- nmpc_casadi/adapt_nmpc.py nmpc_casadi/ac_ablation.py nmpc_casadi/test_ac_ablation.py
    cd nmpc_casadi
    python -m unittest -v test_reward_r1 test_critic_huber_r2a test_adapt_frozen test_heldout_reference test_cad_eval_reference test_ac_ablation

Do NOT change either training reference (\`xref_states_smoothed.csv\`,
\`xref_states_controller.csv\`) or the pretrained checkpoint.
\`baseline.py\` and \`rl_nmpc.py\` are unchanged.

## Smoke test (3 seconds, nominal only)

    $ck = 'D:\GitHub\RL_NMPC\Code\experiment_archive\SELECTED_R3A_HUBER_LR003\checkpoint_best_tracking.npz'
    $cad = 'D:\GitHub\RL_NMPC\Code\experiment_archive\cad_refs\xref_states_cad_test.csv'
    Test-Path $ck
    Test-Path $cad
    python adapt_nmpc.py --track-csv $cad --checkpoint $ck --output ..\..\experiment_archive\eval_cad_ablation_smoke --nominal-only --seconds 3 --rollout-steps 16 --actor-lr 0.001 --critic-lr 0.001 --seed 0 --ablation-components

Expect \`comparison.csv\` with five nominal rows:
fixed, frozen, actor_only, critic_only, both; their solver success, logged
update counts and learning changes must be sensible. The 3-second smoke
is *not* a tracking-quality test, because it does not traverse the full track.

Check:

    $dir = '..\..\experiment_archive\eval_cad_ablation_smoke'
    Import-Csv "$dir\comparison.csv" | Select-Object scenario,controller,updates,actor_unchanged_from_checkpoint,critic_unchanged_from_checkpoint | Format-Table -AutoSize
    Import-Csv "$dir\ablation_sampled_pairing.csv" | Format-Table -AutoSize

Actor-only must show \`critic_unchanged_from_checkpoint=True\`; Critic-only
must show \`actor_unchanged_from_checkpoint=True\`. All sampled prefix
comparisons should show both matched fields True.

## Full evaluation (150 seconds per controller/scenario)

Use an **empty new output** folder:

    python adapt_nmpc.py --track-csv $cad --checkpoint $ck --output ..\..\experiment_archive\eval_cad_ablation_r3a --scenario-set physical --seconds 150 --rollout-steps 16 --actor-lr 0.001 --critic-lr 0.001 --seed 0 --ablation-components --include-frozen-deterministic

The complete run evaluates 6 controllers x 4 physical plant scenarios = 24
rollouts; it does not retrain the checkpoint or write to training data.

Results:
- \`comparison.csv\`: tracking, heading, reward, solver and timing;
- \`pairwise_learning_effect.csv\`: Actor-only, Critic-only, both vs Frozen;
- \`ablation_qfpsi_summary.csv\`: \`qf_2\`, \`q_2\`, \`r_1\`, CTE by mode/scenario;
- \`ablation_sampled_pairing.csv\`: matched first 16 steps across all sampled modes;
- \`{scenario}_{controller}.csv\`: full 71-ms-cycle simulation log;
- \`{scenario}_{controller}_updates.csv\`: per-update loss/gradient diagnostics;
- \`{scenario}_{controller}_adapted_checkpoint.npz\`: adapted network state;
- \`reference_manifest.json\`: CAD SHA256 and separation from training.

Legacy Both-Online retains its original \`{scenario}_ac_updates.csv\` and
\`{scenario}_ac_adapted_checkpoint.npz\` filenames.

## Interpretation (pre-registered, no hyperparameter changes yet)

- Actor-only worse, Critic-only ~ Frozen: Actor/Q/R updates are suspects.
- Critic-only worse, Actor-only ~ Frozen: Critic/Qf/value updates are suspects.
- Both worse while either alone is stable: interaction between updates is likely.
- All Online arms worse: possible general online update instability or objective mismatch.
- Critic-only worse and \`qf_2\` increases: consistent with the Qf_psi hypothesis, **not proof**.
  Next controlled run should hold only \`Qf_psi(s)\` at its pretrained state-dependent value
  while allowing the other Critic weights to adapt.

Compare CTE *and* heading RMSE, reward, peak CTE, solver fallback and timing.
Use a fresh independent track and several random seeds to validate any
selected modification after this diagnostic CAD set.
