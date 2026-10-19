# Step 1 — Critic TD-Huber unchanged, safe-range experiment on Qf_psi

This is an **evaluation-only diagnostic** following the paper's admissible
weight/projection principle (Remark 5 and Algorithm 1, pages 4-5). It is
NOT a stability proof: no terminal invariant set or terminal Lyapunov
decrease guarantee has been checked, and it should NOT be called
a "certified safe controller".

## Scientific question

Earlier tests on a diagnostic AutoCAD route (~10.704 m):
- Frozen Actor+Critic pretrained: ~1.238 cm mean CTE over 4 plant conditions
- Ordinary Online: ~1.852 cm
- Online Critic with Qf_psi FUNCTION held also during learning: ~1.036 cm

However the last intervention changed BOTH the solver heading terminal weight
and the Critic learning gradient. Step 1 changes ONLY the SOLVER INPUT.
Critic TD-Huber, value function, target bootstrapping, gradients, Adam state,
Actor learning, reward and checkpoint are untouched. This isolates the
control-interface effect from the gradient-path effect.

Let F0(s) be the pretrained Critic output for Qf_psi evaluated on the
CURRENT state s (not a single globally frozen scalar).
Let Fc(s) be the currently adapted Critic output.
Apply:

    Fapplied(s) = clamp(Fc(s),
        (1 - epsilon) * F0(s),
        (1 + epsilon) * F0(s))

Only Qf_psi gets projected; other five Qf weights, Q, R retain their
unmodified outputs. The original Critic still learns from its own
UNPROJECTED value function and TD-Huber target. A 0%-band is NOT identical
to the previously tested functional hold inside Critic, because this test
does NOT block TD gradients through Qf_psi.

This is a bounding/projection experiment, not offline-certified Theta.
The admissible bounds epsilon need further robustness and stability checks.

## Controlled comparison

With --qfpsi-trust-frac 0.10 and --ablation-components, controller set
consists of 7 arms:
- fixed_nmpc
- ac_nmpc_frozen (sampled, same saved pretrained RNG)
- ac_nmpc_actor_only
- ac_nmpc_critic_only (unbounded Qf_psi except original broad net bounds)
- ac_nmpc_critic_qfpsi_trust (same Critic learning, solver-only trust)
- ac_nmpc_online (original both-update mode)
- ac_nmpc_online_qfpsi_trust (same both learning, solver-only trust)

Use --include-frozen-deterministic to add the 8th arm. The standard
--isolate-qfpsi mode may also be enabled to compare learning-gradient
masking with solver-only projection, but do this in a later diagnostic run.

Each arm independently reloads the original checkpoint. All sampled
arms should match weights and CTE in the first rollout batch (16 steps),
before their first update, under common Gaussian seed.

## PowerShell setup

In the existing evaluation worktree, preserve user modifications before
overwriting tracked files:

    cd D:\GitHub\RL_NMPC\Code\nmpc_rl_frozen_online
    git status --short
    git fetch origin

    Copy-Item .\nmpc_casadi\adapt_nmpc.py D:\GitHub\RL_NMPC\Code\experiment_archive\adapt_nmpc_before_trust.py

Fetch only the evaluation script and new helper/tests:

    git restore --source=origin/experiment/frozen-vs-online-ac-nmpc -- nmpc_casadi/adapt_nmpc.py nmpc_casadi/qfpsi_trust_region.py nmpc_casadi/test_qfpsi_trust_region.py
    cd nmpc_casadi

Unit tests (nothing here trains a network or overwrites prior runs):

    python -m unittest -v test_reward_r1 test_critic_huber_r2a test_adapt_frozen test_heldout_reference test_cad_eval_reference test_ac_ablation test_qfpsi_isolation test_sampling_seed test_qfpsi_trust_region

Define inputs:

    $ck  = 'D:\GitHub\RL_NMPC\Code\experiment_archive\SELECTED_R3A_HUBER_LR003\checkpoint_best_tracking.npz'
    $cad = 'D:\GitHub\RL_NMPC\Code\experiment_archive\cad_refs\xref_states_cad_test.csv'
    Test-Path $ck
    Test-Path $cad

Smoke test first (3 simulated seconds, nominal only, 0% clamp). The output
directory MUST be new/empty:

    python adapt_nmpc.py --track-csv $cad --checkpoint $ck --output ..\..\experiment_archive\eval_cad_trust0_smoke --nominal-only --seconds 3 --rollout-steps 16 --actor-lr 0.001 --critic-lr 0.001 --seed 0 --qfpsi-trust-frac 0.00 --ablation-components

Check:

    $dir = '..\..\experiment_archive\eval_cad_trust0_smoke'
    Import-Csv "$dir\comparison.csv" | Select-Object controller,updates,qfpsi_solver_only_trust,qfpsi_trust_frac,adapted_checkpoint_saved | Format-Table -AutoSize
    Import-Csv "$dir\ablation_sampled_pairing.csv" | Select-Object controller,matched_weight_prefix,matched_cte_prefix | Format-Table -AutoSize
    Import-Csv "$dir\nominal_ac_nmpc_online_qfpsi_trust_trust_audit.csv" | Select-Object -Last 3 qfpsi_pretrained,qfpsi_raw_candidate,qfpsi_applied,qfpsi_applied_ratio,qfpsi_projected | Format-Table -AutoSize

Expect 7 rows, 5 paired comparisons, matched initial weights/CTE,
2 solver-only-trust rows flagged True and adapted_checkpoint_saved=False.
For eps=0, qfpsi_applied_ratio should remain exactly 1 at every step.
The raw candidate may change and should NOT be forced to 1 by gradient
masking. The *_trust_audit.csv gives the full per-step trace.

If the smoke succeeds, first run a NOMINAL-only FULL ROUTE for the 0% band
before extending to the 4 plant scenarios or nonzero bands:

    python adapt_nmpc.py --track-csv $cad --checkpoint $ck --output ..\..\experiment_archive\eval_cad_trust0_nominal --nominal-only --seconds 150 --rollout-steps 16 --actor-lr 0.001 --critic-lr 0.001 --seed 0 --qfpsi-trust-frac 0.00 --ablation-components

Then try 5% and 10% only in NEW directories (never overwrite an
existing run), and require independent-seed robustness checks later.
For the correct independent Gaussian seeds, use --sampling-seed N;
merely changing --seed does NOT change the checkpoint-restored RNG.

## Crucial interpretation

If 0% solver-only trust recovers Frozen-like CTE, the main benefit is
likely due to restricting the applied NMPC terminal heading weight.
If 0% solver-only trust does NOT recover, while the old functional
gradient-held test did, then modifying the Critic learning path is
itself important.

A clipping projection alone does not ensure closed-loop safety.
Future work needs offline screening of the admissible set, terminal
invariance/stability analysis, and a monitored fallback to Frozen.

The diagnostic CAD has informed method selection. A separately held-out
new AutoCAD route and multiple sampling seeds are required for final
generalization claims. We should not tune to this route and then report
it as an untouched test.
