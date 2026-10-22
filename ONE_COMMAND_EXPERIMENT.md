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
