# 100-EPISODE FIRST LONGER TRAIN experiment (gamma=0.90)

**Research purpose:** The one-episode gamma=.90 smoke completed
1048 steps and reached end, solver success 100%, but deterministic
frozen inference was nearly identical before/after training:
observed start discounted cost 14.500389665 (untrained) vs
14.500396199 (trained), not an improvement. A one-episode
smoke can confirm wiring, NOT learnability. The user explicitly
requests a multi-episode learning trial.

## Protocol: 100 episodes, same training settings

Keep **gamma=.90** (user selected), fixed reward Eq.(4)
`Qe=[20,20,100,20,600,600]` and
`Re=[50,100]`, original 20% exploratory box and candidate-2
initial Q/R/P, Actor Gaussian initial sigma=.10
and sigma [.02,.30]. Plain SGD Actor learning rate 1e-6,
projected TD(0) six-P Critic learning rate .001. These last
numbers are EXPERIMENTAL controls inherited from the 10-step
smoke, NOT proven effective or specified by source paper.

Only the pinned TRAIN reference is read. No TEST for training
or checkpoint selection.

### 1. From nmpc_casadi PowerShell

```powershell
git status --short
git pull --ff-only

python -m unittest -v test_evaluate_training_curve test_frozen_actor_critic_eval

python train_actor_critic.py `
  --reference .\reference_split\train_xref.csv `
  --screen-manifest .\results_screen\full_20pct\manifest.json `
  --reward-audit-summary .\results_reward_audit\candidate_qe_re_g090\summary.json `
  --output .\results_rl\gamma090_train100_seed0 `
  --acknowledge-empirical-only `
  --reward-candidate legacy `
  --gamma 0.90 `
  --sigma-lower 0.02 --sigma-initial 0.10 --sigma-upper 0.30 `
  --actor-lr 0.000001 --critic-lr 0.001 `
  --episodes 100 --steps-per-episode 1150 `
  --seed 0 --ey 0.05 --heading-deg 0
```

The 27-case reward audit for gamma=.90 already exists; no need to
run it again. Training writes a JSON line after each episode and
updates `training_history.csv`; inspect it while training.
Training can take significant wall-clock time (~100,000 NMPC
solves), particularly on Windows. **Do not re-run with same
output directory:** the driver rejects overwriting evidence.
The current driver has NO true resume mode: if execution stops
at episode k, only 1..k are saved, and re-run needs a new
output directory and restarts from initial parameters. A resume
feature requires restoring exact Gaussian RNG state and is NOT
silently approximated.

### 2. Frozen evaluation of learning curve (multiple checkpoints)

After all 100 episodes complete, run frozen checkpoints
at 1,10,25,50,100 (i.e. internal zero-based 0,9,24,49,99):

```powershell
python evaluate_training_curve.py `
  --train-root .\results_rl\gamma090_train100_seed0 `
  --reference .\reference_split\train_xref.csv `
  --screen-root .\results_screen\full_20pct `
  --output .\results_rl\gamma090_eval100_seed0 `
  --epochs 1 10 25 50 100 `
  --max-steps 1150 --all-train-conditions
```

This deliberately replays deterministic frozen control on all
three existing TRAIN initial conditions (same path but including
two start conditions that were not used to update Actor/Critic).

Each snapshot compares exactly FIVE controllers:

1. Untrained deterministic Actor and initial direct P
2. Trained deterministic Actor and current frozen P
3. Fixed NMPC candidate 2
4. Strong tuned fixed NMPC
5. Stage-lower/terminal-upper fixed NMPC (competitive TRAIN comparator)

The last three comparators reuse existing empirical screening
logs (not re-solved) and recompute the EXACT same fixed Qe/Re
reward. All conditions are TRAIN; this is NOT new held-out TEST.

### 3. How to assess learning

`results_rl\gamma090_train100_seed0\training_history.csv`
shows one row per episode. Check:

- Mean reward trend, and how noisy it is; not just one episode.
- RMS TD error trend (it does not necessarily monotonically decay
  in online stochastic learning).
- Solver success, reached_end, time truncation.
- CTE trend and errors from the per-step episode CSVs.
- Critical: the 6 P values and both Gaussian `mu` and
  `sigma` heads must demonstrably update. Do NOT infer
  learning from stochastic Q/R sampling alone.

`results_rl\gamma090_eval100_seed0\learning_curve_comparison.csv`
is the frozen checkpoint-vs-baseline comparison. The nested
`epoch_0001/`, ..., `epoch_0100/` folders preserve each
individual comparison and vehicle trace. The file
`learning_curve_summary.json` includes start/end reward and
TD diagnostics.

**A meaningful positive result:** frozen-trained improves
vs frozen-untrained as epochs increase under matching
conditions, reaches the track end with high solver success,
and ideally outperforms BOTH the well-tuned original fixed
NMPC and strongest screened fixed comparator on an explicitly
fixed reward plus safety/tracking metrics. One seed and TRAIN
can suggest improvement, not prove generalization or
closed-loop stability.

If 100 episodes still do not produce real improvement,
report the failure transparently. The prior return analysis
already showed fixed P box fitted cost-to-go poorly, and
this run's LR and P-box may suppress real learning. Do NOT
claim that extending to 1000 episodes fixes it automatically;
diagnose gradients/relative changes and decide a separate
hyperparameter experiment.

### 4. Files to upload for next round

- `gamma090_train100_seed0/training_history.csv`
- `gamma090_train100_seed0/run_config.json`
- `gamma090_eval100_seed0/learning_curve_comparison.csv`
- `gamma090_eval100_seed0/learning_curve_summary.json`
- `gamma090_train100_seed0/checkpoint_ep0099.npz` (optional)
- `gamma090_train100_seed0/episode_0099.csv` (optional)

If evaluation is interrupted, the script writes
`learning_curve_comparison.csv` incrementally after each
fully evaluated checkpoint but will not overwrite old
nonempty output folders. Preserve these results for review.

## Mathematical caveats

One-hundred stochastic-policy training episodes share the
SAME TRAIN reference and one starting condition; replay at
two additional TRAIN offsets checks robustness only. Frozen
hold-out TEST must be kept untouched until experiment
settings/checkpoint are finalized. The 27 fixed-weight
screening results do not certify a time-varying actor
policy's recursive feasibility, stability or positive
terminal invariant set. The critic remains exactly six
physical P parameters, and the Actor remains the approved
two-headed Gaussian 17-64-64 network.
