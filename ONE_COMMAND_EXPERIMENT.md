# ONE COMMAND — 100 episodes, auto-evaluate, save tracking/Q/R/Qf images

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

## B. Run

In PowerShell within `nmpc_casadi`:

```powershell
git pull --ff-only
python run_experiment.py
```

**That is all.** There are no required command-line flags and no
per-episode output folders.

Current `experiment_config.py` defaults are explicit:
`EPISODES=100`, `STEPS_PER_EPISODE=1150`, `GAMMA=.90`,
`SEED=0`, `SIGMA_MIN=.02`, `SIGMA_INIT=.10`,
`SIGMA_MAX=.30`, `ACTOR_LR=1e-6`,
`CRITIC_LR=1e-3`, `WEIGHT_BOX_MARGIN=.20`,
`CHECKPOINT_EPISODES=(1,10,25,50,100)`.

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
1, 10, 25, 50, 100. A complete `training_history.csv` is
maintained every episode.

After episode 100, the runner freezes each selected checkpoint
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

**Caveats:** This first 100-episode run uses one seed, one TRAIN
path and one initial condition. It is NOT held-out TEST,
and the exploratory parameter box is NOT a certified terminal
stability region. If the Critic hardly changes, inspect learning
rate and box bounds instead of claiming 100 episodes proved
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
The 100-episode loop is invoked only by `python run_experiment.py`.
