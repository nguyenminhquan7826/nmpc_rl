# ONE COMMAND — gamma=.90, 10-episode componentwise Critic LR pilot, auto-evaluate, save tracking/Q/R/Qf images

The simplified experiment was requested to avoid repeatedly typing
`--screen-manifest`, `--reward-audit-summary` and numerous result paths.
**All user-editable experiment parameters live in
`nmpc_casadi/experiment_config.py`.**

The old CLI utilities continue to exist for reproducibility and optional
scientific audits, but this new entry point NEVER requires their
manifest or summary file paths. It DOES re-verify the exact TRAIN
SHA256, never uses TEST, and still enforces the previous Q/R/Qf
numerical exploratory box. This does NOT confer a stability theorem.

## A. Folder layout seen by the simplified runner

From `nmpc_casadi`:

```text
nmpc_casadi/
|-- experiment_config.py      # change all user-facing settings HERE
|-- run_experiment.py         # one command
|-- ...                       # other Python code, unchanged
|-- reference_split/          # ONE trajectory folder (already existed)
|   |-- train_xref.csv        # pinned original TRAIN CSV, unchanged
|   |-- _nmpc_initial_reference_tail.csv
|   |-- _rl_validated_train_copy.csv
|   '-- _rl_prediction_tail.csv
|-- results/                  # ONE results folder, flat (no subfolders)
|   |-- run_config.json
|   |-- training_history.csv
|   |-- episode_0000.csv      # selected episodes only
|   |-- episode_0009.csv
|   |-- episode_0024.csv
|   |-- episode_0049.csv
|   |-- episode_0099.csv
|   |-- checkpoint_ep0000.npz # selected checkpoints only
|   |-- checkpoint_ep0009.npz
|   |-- checkpoint_ep0024.npz
|   |-- checkpoint_ep0049.npz
|   |-- checkpoint_ep0099.npz
|   |-- eval_trained_episode_0001.csv
|   |-- ...                   # other frozen comparator XY, Q/R/Qf logs
|   |-- comparison.csv
|   '-- experiment_summary.json
'-- images/                   # ONE images folder, flat (no subfolders)
    |-- trajectory_tracking.png
    |-- q_mean_episodes.png
    |-- r_mean_episodes.png
    |-- qf_final_episodes.png
    |-- training_reward_td.png
    '-- frozen_tracking_comparison.png
```

Previously existing folders (e.g. `results_screen`, `results_rl`,
`results_critic_audit`, `sweep_results`) are **historical
evidence**. The simplified run does not read them or recreate
`sweep_results` subfolders, and NEVER deletes earlier data. You
may archive older directories manually after reviewing them; this
experiment does not silently remove research evidence.

All data under `reference_split` is a TRAIN reference or derived
nonphysical prediction tail, NOT an additional TEST track.

## Best model and its actual training trajectory

Run `python run_experiment.py` as usual. A new best completed training
episode (solver success at least 99%, then lowest mean CTE) immediately
saves `checkpoint_best_train.npz`, `episode_best_train.csv` and
`best_train_selection.json` in the configured results folder, even when
the episode is not in the scheduled checkpoint list.

The CSV now contains actual pre-action `X_m`, `Y_m`, matching
`X_ref_m`, `Y_ref_m`, `t_s`, and post-action `X_next_m`, `Y_next_m`.
At experiment completion, `best_train_episode.png` in the configured
images folder overlays this actual trajectory on the reference, including
the final vehicle position, alongside CTE and reward panels. Older logs
without XY retain the CTE/reward plot; no trajectory is reconstructed.

The saved model is the model **after the last update** of that episode.
The training trajectory includes exploration and online updates, so it
is not a frozen replay of that checkpoint. The existing
`checkpoint_best_frozen.npz`, `eval_best_frozen.csv` and
`best_frozen_trajectory.png` report the best eligible deterministic
evaluation among checkpoints actually evaluated. If no episode qualifies,
no best model or trajectory is claimed. Learning rates are unchanged.

## Before the first run: pinned TRAIN file is local and gitignored

The `reference_split/` folder is deliberately excluded from Git
to keep research CSV inputs private. Thus a fresh cloned repo or
switching to a separately checked-out source directory does **not**
install `train_xref.csv` by using `git pull`.

Starting with this revision, `python run_experiment.py` performs a
bounded, read-only search of nearby project checkouts and saved
`cad_eval_source_xref.csv` copies from previous training runs.
It restores a local copy into
`reference_split/train_xref.csv` **only if its SHA256 is exactly**
`afa67ac24ef61b591b1a16498bcb1ee5e521452cf0f8036de00923d062940b98`.
It will never synthesize TRAIN from the two tracked historical
`xref_states_*.csv` files, nor read `test_xref.csv`.
It refuses to overwrite an existing (even corrupted) TRAIN CSV.

If your original TRAIN lies outside the nearby checkout search,
set the single Python variable in `experiment_config.py`:

```python
TRAIN_CSV_SOURCE = Path(r"D:\\some-existing-old-checkout\\reference_split\\train_xref.csv")
```

This configuration variable is optional; no CLI arguments.
Another option is to **copy the original** pinned file manually into
the new checkout's `reference_split/` folder. Do not rename
`xref_states_smoothed.csv` as train data.

Windows PowerShell can find the original location if needed:

```powershell
Get-ChildItem "D:\\GitHub\\RL_NMPC" -Recurse -File -Filter train_xref.csv -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty FullName
```

Once found, verify the SHA before reuse:

```powershell
(Get-FileHash "FULL_PATH_TO_TRAIN_XREF_CSV" -Algorithm SHA256).Hash
```

If no matching local source exists, 100-episode learning
**must not** start; re-create TRAIN from the original sanctioned
TRAIN/TEST split procedure, with matching provenance, rather
than guessing a different file.

## B. Run

In PowerShell within `nmpc_casadi`:

```powershell
git pull --ff-only
python run_experiment.py
```

**That is all.** There are no required command-line flags and no
per-episode output folders.

Current `experiment_config.py` defaults are explicit:
`EPISODES=10`, `STEPS_PER_EPISODE=1150`, `GAMMA=.90`,
`SEED=0`, `SIGMA_MIN=.02`, `SIGMA_INIT=.10`,
`SIGMA_MAX=.30`, `ACTOR_LR=1e-6`,
`CRITIC_LR=1e-2`, `WEIGHT_BOX_MARGIN=.20`,
`CHECKPOINT_EPISODES=(1,5,10)`, 
`CRITIC_P_LR_FACTORS=(1.23,.86,.25,.41,1.84,1.85)` and
`ACTOR_OUTPUT_LR_FACTORS=(1.,)*8`.

The fixed task reward remains Eq.(4), with legacy experimental
`Qe=[20,20,100,20,600,600]`,
`Re=[50,100]` in `audit_training_reward.py`; this is independent
of adaptive Actor Q/R. The approved candidate-2 initial weights
are in `training_initialization.py`. No user has to pass these
as paths or CLI arguments. In the future, altering the research
reward/bounds requires a separately documented experiment.

The prior 27/27 empirical full screening informed the
`screening_box(.20)` constants, but the simplified run **does
not reload or re-verify** any archived screening manifest or
old offline reward diagnostic. Run metadata records this weaker
provenance explicitly. All runtime reward and TD values are still
computed from live NMPC transitions exactly as before.

## C. Plots and comparators

Every episode: collect *actual sampled* Q/R means, six learned
Critic physical P (the stage NMPC terminal Qf diagonal) at episode
end, reward and TD. To avoid 200+ log/checkpoint files, detailed
per-step CSV and checkpoints are saved only after episodes
1, 5 and 10 for the Critic-LR pilot. A complete `training_history.csv` is
maintained every episode.

After episode 10, the runner freezes each selected checkpoint
(no stochastic Actor sampling, no gradients) and compares against:

- Initial untrained deterministic Actor and P;
- Fixed NMPC with candidate-2 Q/R/P;
- Original tuned strong fixed NMPC;
- Strong sampled stage-lower/terminal-upper fixed NMPC.

The three fixed controllers are **rerun using the same live
NMPC solver and plant**, so old `results_screen` output
folders are NOT needed for comparison. All plots use actual
logged vehicle `X_m,Y_m` and reference `X_ref_m,Y_ref_m`,
not synthetic trajectories.

A separate PNG presents trajectory tracking; three others show
Q, R, Qf vs TRAIN episode, plus reward/TD and frozen CTE curves.
`comparison.csv` uses the same fixed reward for every controller.

**Caveats:** This pilot run uses one seed, one TRAIN
path and one initial condition. It is NOT held-out TEST,
and the exploratory parameter box is NOT a certified terminal
stability region. If the Critic hardly changes, inspect learning
rate and box bounds instead of claiming the pilot proved
learnability.

## D. Existing output safety

The runner refuses to overwrite nonempty `results/` or
`images/`, or existing generated RL trajectory/tail names.
No automatic deletion or resume logic is provided. Before a
new *independent* experiment, manually archive completed
`results/` and `images/` (e.g., zip them), and move generated
`_rl_*.csv` derivative trajectory files aside if necessary.
Do NOT delete the original pinned `train_xref.csv`.

Note that merely running `git pull` does NOT start training.
The training loop is invoked only by `python run_experiment.py`.

## Pilot restart after a completed prior 100-episode experiment

The previous completed TRAIN files in `results/` and `images/` must
be preserved before any new run: this runner *refuses to overwrite*
nonempty output folders. Move those two existing directories into
an archival path **outside the checkout**, e.g. in your Documents
folder, and confirm they contain your old CSV/PNG before running
`python run_experiment.py`. Do not move, rename or delete the pinned
`reference_split/train_xref.csv`. Do not modify TEST.

Pilot purpose: compare changing ONLY Critic physical-P per-component
learning rates against the original actor/critic 100-episode baseline.
Actor still learns with its previous rate of 1e-6; only the Critic's
learning rates change. No proof of optimality or stability is inferred.

## Online adaptation: keep Actor and Critic learning after training

`RUN_ONLINE_ADAPTATION = True` in `experiment_config.py` adds an online
rollout after frozen comparison, starting from the best frozen checkpoint
(or best TRAIN checkpoint if no frozen checkpoint qualifies).
Set it to `False` to disable. Frozen comparison is preserved separately.

Each step samples Gaussian Actor actions, applies Q/R and current Qf to
NMPC, then updates all Actor NN parameters and the six direct Critic P
parameters using the same pre-update TD. The next step uses updated
models. This is online learning on TRAIN, not independent frozen testing.
The Critic has no neural network. Projection at a bound may keep P flat.

Files in `RESULTS_DIR/online_adaptation/`:
- `online_episode.csv`: applied q/r/p, p_after, TD, Actor/Critic parameter
  step norms, XY, solver diagnostics and task/learning rewards.
- `checkpoint_after_online.npz`: final updated model; original untouched.
- `online_summary.json`: metrics, source checkpoint, adaptation seed/mode.
- `online_weights.png`, `online_trajectory.png`, `online_qf_change.png`.

To adapt an existing checkpoint without retraining, from nmpc_casadi:

```powershell
python online_adaptation.py --checkpoint results_v1/checkpoint_best_frozen.npz --reference reference_split/train_xref.csv --output online_results_v1 --max-steps 1150 --seed 0
```

Replace folder names with your paths. Output must be empty. Bounds, gamma,
LRs, factors and penalties come from checkpoint metadata, not the current
margin config. Task reward is separate from penalized learning reward.
ONLINE_ADAPTATION_SEED starts a new sampling stream, not exact RNG resume.
The final model is saved without claiming it is a best checkpoint.
The reference must match pinned TRAIN; no held-out TEST tuning is done.
Full online runs require that local TRAIN CSV, which is not stored in Git.

## Signed TD diagnostics for every training episode

New runs write mean_td, rms_td, mean_abs_td, std_td, negative_td_pct,
positive_td_pct and mean_td_e2_0 through mean_td_e2_5 in training_history.csv
and the JSON console record for EVERY episode, regardless of --save-at.
mean_td_e2_i is mean(delta * e_i^2), before LR scaling and projection.
It describes the raw component update direction, not the actual bounded step.

training_reward_td.png now has three separate panels: mean task reward,
signed mean TD with a zero reference, and RMS TD. No sign is inferred from
RMS. Old histories lacking mean_td explicitly display it as unavailable.
Use a new results/images directory when retraining to preserve prior outputs.
Learning rates, bounds, rewards and update formulas are unchanged.

## Experimental full gradient of sampled squared TD loss

CRITIC_UPDATE_MODE = "residual_gradient" is enabled for the next comparison.
Use "semi_gradient" to restore the previous method. The training CLI exposes
--critic-update-mode with the same two choices; its default remains the old
semi-gradient for backward compatibility. New default output directories are
results_residual_gradient and images_residual_gradient. Do not reuse nonempty
outputs. LRs, gamma, bounds, Actor update and reward are unchanged.

For delta = r + gamma V_P(e_next) - V_P(e), V_P(e) = -sum(P_i e_i^2):
- semi_gradient: g_i = delta * e_i^2
- residual_gradient: g_i = delta * (e_i^2 - gamma * e_next_i^2)
- true terminal: both use g_i = delta * e_i^2 (no bootstrap)
- time-limit truncation: retain bootstrap in both methods.
P_new = project(P_old - component_LR * g). Transition data are held fixed;
there is no derivative through NMPC, the plant or the Actor. The Actor uses
the same pre-update delta in either mode.

This minimizes sampled squared TD loss, not an unbiased gradient of squared
expected Bellman error in stochastic dynamics/policies (double-sampling
issue). It is an experimental alternative, not a guarantee of better control
or convergence to zero TD. Near-identical successive errors may make this
gradient small; do not increase rates until actual gradients are inspected.

Checkpoint/run_config metadata records critic_update_mode. Online adaptation
inherits the checkpoint mode; old checkpoints with no field keep semi-gradient.
Detailed episode logs add critic_gradient_0..5, the actual loss derivatives
before LR/projection; history adds mean_critic_gradient_0..5. Existing
mean_td_e2_i still means delta*e_i^2 and is not the residual-gradient update.
Compare signed mean TD, RMS, actual gradients, P drift/bound hits and frozen
tracking under the same initial seed and episode budget. Freeze parameters
for scoring; online-adaptation results remain a separate learning rollout.

## Five-mode checkpoint evaluation on a supplied route

No retraining is required. From nmpc_casadi:

```powershell
python evaluate_controller_modes.py --checkpoint results_v14/checkpoint_best_frozen.npz --reference reference_split/test_xref.csv --output evaluation_v14 --max-steps 2000 --seeds 0 1 2 --sampling-control
```

Replace paths with existing files. The supplied route uses the project's
10-column XREF CSV format. It is validated and copied unchanged, with a
separate prediction tail. A different SHA256 from TRAIN is recorded but does
not itself prove a route is an independently designed held-out test.

Modes: fixed (initial Q/R/P from checkpoint metadata), frozen learned (deterministic mean),
actor_adapt (sampled actions, only Actor parameters update), critic_adapt
(deterministic Actor mean, only P updates), full_adapt (sampled actions,
both update). Optional frozen_sampled uses sampling with neither model
updating, to distinguish exploration effects from learning. These experiments
use the existing nominal plant; model-parameter disturbances are not injected.

Each mode and seed independently reloads the same checkpoint, keeps its
saved bounds/gamma/LRs, and resets environment, warm-start and reference
manager. Nothing is carried between modes. --ey and --heading-deg set shared
initial offsets. Checkpoint source and route hashes are recorded. Adaptation
modes learn on the route and must be reported separately from frozen scores;
do not tune settings on a route you intend to call held-out final TEST.

Outputs: comparison.csv (all runs), summary_by_mode.csv (mean/std, including
failed runs), evaluation_config.json (reproduction settings), seed_N/
mode_comparison.png (XY and CTE), and seed_N/MODE/trajectory.csv. Adaptation
modes also save checkpoint_after.npz, preserving the original checkpoint.
Report completion/failure and solver success before interpreting return and
CTE: early failure can artificially improve total reward. Deterministic modes
repeat identically across seeds without external noise; stochastic modes
require multiple seeds. Existing post-capture metrics fall back to the whole
log if a run never reaches the capture threshold; all-route CTE is separately
reported. Full loop timing includes observation/plant/reference processing,
Actor inference, solver, update diagnostics and online learning (not plot/file
I/O); it is host timing, not embedded hardware timing. Disabled learner
components evaluate gradients diagnostically without installing their updates.

This runner does not replace run_experiment.py or its original frozen scoring.


## Simplified evaluation figures and initial fixed baseline

Fixed now uses checkpoint initial_Q, initial_R and initial_P, never GOOD_NMPC.
Default evaluation output contains only three PNGs per seed: trajectories.png
(all requested modes), cte_mean_max.png (absolute CTE over all logged steps,
including initial transient), full_adapt_weights.png (applied Q/R/Qf).
Incomplete runs are marked on comparison figures. Full-adapt weights use
pre-update P applied to each NMPC step, not post-update diagnostics.

Run five modes with images only, from nmpc_casadi:

```powershell
python evaluate_controller_modes.py --checkpoint results_v11_best/checkpoint_best_frozen.npz --reference test_reference/test_xref.csv --output evaluation_v11_initial_fixed --max-steps 2500 --seeds 0
```

Optional --sampling-control adds the frozen sampled mode. --save-data opts in
to the earlier CSV/settings/checkpoint outputs; without it no model or numeric
files are written. Console still prints the run summaries. Use a new directory
to avoid mixing old GOOD_NMPC fixed results with the new initial baseline.
The older output listing above applies only when --save-data is supplied.

## Start adaptation at the same nominal weights as Fixed

Default --adapt-start initial_heads retains the trained Actor hidden layers,
but resets W_mu/W_sigma to zero and b_mu/b_sigma to their original calibrated
values. Thus deterministic Actor Q/R initially equal checkpoint initial_Q and
initial_R for every observation. Critic P resets to initial_P. All adaptation
modes and the optional frozen_sampled control use independent identical reset
policies, not the trained controller. Frozen mode remains the original trained
checkpoint and is labelled frozen as before. Fixed uses initial Q/R/P.

Actor-only therefore holds initial_P fixed. Critic-only holds the reset Actor
fixed (deterministic outputs equal initial Q/R). Full adapt updates both from
the reset policy. This is relearning from a reset output policy with transferred
features, not unchanged-checkpoint adaptation; report that distinction.
Sampled actions need not equal the nominal initial Q/R on the first step:
the latent Gaussian mean is calibrated to the initial weights, while sampling
adds exploration. No changes are made to the checkpoint on disk.

Use --adapt-start fresh to restore ALL Actor parameters to original seeded
initialization, with initial_P. This evaluates online learning from scratch.
Use --adapt-start trained to restore the earlier adaptation from the trained
NN/P. All runs still use saved learning rates/bounds unless explicitly overridden
by your load_checkpoint configuration. Use fresh output directories.
