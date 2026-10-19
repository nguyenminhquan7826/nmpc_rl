# Phase 2B — Reward-driven learning of independent terminal Qf_psi

**Evaluation-only, no change to training checkpoint or core rl_nmpc.py.**

## Scientific motivation and comparison

Empirical nominal AutoCAD track results (same checkpoint/RNG):

| Controller | Mean CTE (cm) | Max CTE (cm) | Heading RMSE (rad) | Reward |
| --- | ---: | ---: | ---: | ---: |
| Frozen | 1.243 | 3.920 | .06612 | -891.88 |
| Online original (TD gradients also drive Qf) | 1.865 | 4.544 | .05867 | -933.89 |
| Online solver-input-only Qfpsi Trust 0% | 1.178 | 3.146 | .06616 | -864.86 |
| Split Critic Phase 2A (all Qf frozen) | 1.238 | 3.342 | .06592 | -875.86 |
| Actor only | 1.233 | 3.382 | .06595 | -875.98 |

Phase 2A established that separating Value Critic from all six NMPC Qf
outputs retains nearly the Actor-only performance. Phase 2B investigates
whether terminal weights can be RE-LEARNED FROM CONTROL REWARD without
receiving a TD-Huber Critic gradient.

## Key experimental design

- Independently copy checkpoint 7-output Critic into a 7-output Value
  network and a 6-output Terminal network.
- Value network retains the same n-step TD-Huber loss and internal
  quadratic basis V(s,e) = -b(s) - e^T Qf_value(s)e. These six VALUE BASIS
  outputs no longer drive NMPC's actual terminal weights.
- The original Actor still learns from one-step TD policy gradient,
  with the exact same saved checkpoint and own RNG.
- Terminal network's hidden layers and the five outputs OTHER THAN psi
  are BITWISE frozen. Only output-layer Qf_psi column and bias can change.
  This first pilot is NOT yet full 6-dimensional Qf control learning.
- Terminal Qf_psi latent logit is sampled:
      z ~ Normal(mu_omega(s), sigma_qf**2).
  Candidate Fpsi(s,z) is transformed via sigmoid to the original
  positive bounded terminal multiplier and then projected to
  [(1-eps) F0psi(s), (1+eps) F0psi(s)], F0 from pretrained model at s.
- The score-function update optimizes the actual shared control reward
  through the stochastic Qf policy WITHOUT differentiating through NMPC:

      delta_k = r_k + gamma*(1-terminated_k)*V_phi(s_{k+1},e_{k+1})
                      - V_phi(s_k,e_k)
      A_k = delta_k / max(std_batch(delta), 1)
      L_Qf_pg = -mean_k[A_k * log Normal(z_k; mu_omega(s_k), sigma_qf)]
      L_Qf_trust = mean_k[log(Fpsi_mean(s_k)/F0psi(s_k))**2]
      L_terminal = L_Qf_pg + lambda * L_Qf_trust

  Critic V_phi is evaluated BEFORE its own TD update, and detached from
  terminal gradients. Actor and terminal PG share the SAME pre-update
  one-step TD-based advantage baseline, but their parameters/optimizers
  are distinct. This is reward-driven policy-gradient learning, NOT
  direct TD loss backpropagation into Qf.
- A fresh terminal Adam optimizer (zero moments/t=0) avoids bringing
  TD-Critic optimizer momentum into the new policy-gradient problem.
- The probability term is computed for the UNCLIPPED latent sample,
  which is essential for score-function gradients with a clipped
  downstream action.
- Solver-input clipping is heuristic and does NOT provide a terminal-set,
  Lyapunov stability, or real-time feasibility proof.

## Mandatory paired controls

When --split-qf-policy-gradient is enabled, run three separate modes:

1. ac_nmpc_split_value_frozen_qf:
   Phase 2A, NO terminal sampling, NO terminal-head update
2. ac_nmpc_split_qf_sampled_frozen:
   terminal Qfpsi stochastic sampling/clipping, NO terminal-head update
3. ac_nmpc_split_qf_pg:
   EXACT SAME initial Qfpsi samples/clipping, but terminal-head learns
   after first rollout batch

The last two arms must have **identical complete (Q,R,Qf) and CTE over
the first 16 steps**, as they load identical checkpoints, Actor RNG
and independent Qf RNG. The evaluator writes phase2b_sampled_pairing.csv
and fails if that condition is violated.

The Qf-stochastic modes intentionally do NOT have a matched initial
weight/CTE prefix against ordinary Frozen, since new terminal
exploration occurs from the first step. Do not treat that expected
difference as a broken experimental pairing.

## 1) Update evaluation-only files

    cd D:\GitHub\RL_NMPC\Code\nmpc_rl_frozen_online
    git status --short
    git fetch origin

Backup evaluator before replacing:

    Copy-Item .\nmpc_casadi\adapt_nmpc.py D:\GitHub\RL_NMPC\Code\experiment_archive\adapt_nmpc_pre_phase2b.py

    git restore --source=origin/experiment/frozen-vs-online-ac-nmpc -- nmpc_casadi/adapt_nmpc.py nmpc_casadi/split_terminal_pg.py nmpc_casadi/test_split_terminal_pg.py

    cd nmpc_casadi

## 2) Run unit tests first

    python -m unittest -v test_split_terminal_pg
    python -m unittest -v test_reward_r1 test_critic_huber_r2a test_adapt_frozen test_heldout_reference test_cad_eval_reference test_ac_ablation test_qfpsi_isolation test_sampling_seed test_qfpsi_trust_region test_split_critic_eval test_split_terminal_pg

Expected: previous 57 plus 9 new tests = 66, if every test file has
been checked out. Do not proceed if errors occur.

## 3) Smoke test Phase 2B, Nominal, 3 simulated seconds

    $ck  = 'D:\GitHub\RL_NMPC\Code\experiment_archive\SELECTED_R3A_HUBER_LR003\checkpoint_best_tracking.npz'
    $cad = 'D:\GitHub\RL_NMPC\Code\experiment_archive\cad_refs\xref_states_cad_test.csv'

    python adapt_nmpc.py --track-csv $cad --checkpoint $ck --output ..\..\experiment_archive\eval_cad_split_2b_smoke --nominal-only --seconds 3 --rollout-steps 16 --actor-lr 0.001 --critic-lr 0.001 --seed 0 --ablation-components --split-critic --split-qf-policy-gradient --qfpg-trust-frac 0.05 --qfpg-sigma 0.02 --qfpg-lr 0.0001 --qfpg-reg 0.10

Output folder MUST be NEW and empty; don't overwrite Phase 2A outputs.

Acceptance checks:

    $dir = '..\..\experiment_archive\eval_cad_split_2b_smoke'

    Import-Csv "$dir\comparison.csv" |
      Select-Object controller,updates,split_qf_head_training_enabled,split_qf_head_unchanged,split_qf_head_untouched_components_unchanged,split_qf_head_training_updates |
      Format-Table -AutoSize

    Import-Csv "$dir\phase2b_sampled_pairing.csv" |
      Format-List scenario,pre_update_steps,matched_weight_prefix,matched_cte_prefix

    Import-Csv "$dir\nominal_ac_nmpc_split_qf_pg_qfpg_audit.csv" |
      Select-Object -Last 5 qfpg_pretrained_psi,qfpg_mean_psi,qfpg_candidate_psi,qfpg_applied_psi,qfpg_applied_ratio |
      Format-Table -AutoSize

Expected on Phase 2B learned arm:
- updates > 0 and split_qf_head_training_updates equals updates
- split_qf_head_training_enabled=True
- split_qf_head_unchanged=False (terminal psi output trained)
- split_qf_head_untouched_components_unchanged=True
- both stochastic modes have same complete weight+CTE FIRST 16 steps
- every applied_ratio within [0.95, 1.05]
- no model or checkpoint overwrite.

## 4) Full nominal diagnostic, AFTER smoke passes

    python adapt_nmpc.py --track-csv $cad --checkpoint $ck --output ..\..\experiment_archive\eval_cad_split_2b_nominal --nominal-only --seconds 150 --rollout-steps 16 --actor-lr 0.001 --critic-lr 0.001 --seed 0 --ablation-components --split-critic --split-qf-policy-gradient --qfpg-trust-frac 0.05 --qfpg-sigma 0.02 --qfpg-lr 0.0001 --qfpg-reg 0.10

Compare the two stochastic arms to attribute performance change to
LEARNING, not noise. The unperturbed split Phase 2A is a separate baseline.
Evaluate Mean/Max CTE, heading RMSE, reward, completion, solver success
and wall-time distribution. Multiple independent sampling seeds and new
CAD geometry are required before drawing general conclusions.

This is ONLY a diagnostic. Do not deploy on a physical vehicle; a
pretrained-relative box projection does not certify MPC safety.
