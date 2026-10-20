# Direct-P Actor NN + Critic TD(0) — NMPC

**Branch:** \`experiment/direct-quadratic-td-critic\`  
**Original research repo:** \`nguyenminhquan7826/nmpc_rl\`  
**Full snapshot before removing historical outputs:** \`backup/direct-p-before-cleanup-20261020\`

This branch keeps the **runnable Direct-P experiment**, its imported numerical
model and CAD reference helpers, plots and regression tests. Large old NN-Critic
results, their raw episode logs and unused legacy scripts are removed **from
this branch only**. They remain available in the backup branch or Git history.
Original branches \`main\`, \`experiment/frozen-vs-online-ac-nmpc\`, etc. were
not touched.

## Relevant files

| File | Purpose |
|---|---|
| \`nmpc_casadi/train_direct_p.py\` | Train Actor NN and six positive diagonal terminal coefficients using TD(0) |
| \`nmpc_casadi/eval_direct_p_heldout.py\` | Compare Fixed NMPC / frozen Actor & P / test-time adaptation on held-out track |
| \`nmpc_casadi/direct_p_critic.py\` | Direct six-parameter semi-gradient TD Critic, inherited NN Actor |
| \`nmpc_casadi/plot_direct_p_logs.py\` | Automatically make or regenerate all train/test PNGs from CSV |
| \`nmpc_casadi/rl_nmpc.py\` | Shared Actor NN, environment, reward, Gaussian policy, checkpoint utilities |
| \`nmpc_casadi/baseline.py\`, \`nmpc_casadi/nmpc_casadi.py\` | Vehicle model, reference manager, CasADi NMPC |
| \`nmpc_casadi/adapt_nmpc.py\` | Held-out reference and plant profiles used by evaluator; retains older diagnostic helpers |
| \`nmpc_casadi/qfpsi_isolation.py\` | Transitive import of \`adapt_nmpc.py\`, required until that module is refactored |
| \`nmpc_casadi/test_*.py\` | Direct-P, plotting, reference, reward and adapter regression tests |

See \`nmpc_casadi/DIRECT_P_EXPERIMENT.md\`,
\`nmpc_casadi/DIRECT_P_PLOTTING.md\` and
\`nmpc_casadi/DIRECT_P_HELDOUT_RESULTS.md\` for details.

## Windows PowerShell quick start

Use a virtual environment with Python 3.11–3.13 and install requirements:

\`\`\`powershell
cd D:\GitHub\RL_NMPC\Code\nmpc_rl_direct_p
git pull --ff-only origin experiment/direct-quadratic-td-critic
cd nmpc_casadi
python -m pip install -r requirements.txt
python -m unittest -v test_direct_p_critic test_direct_p_plots
\`\`\`

The independent TRAIN/TEST XREF CSV files are **local user-supplied data**;
the repository does not commit them. For example,
\`reference_split/train_xref.csv\` and \`reference_split/test_xref.csv\` should
already exist in your clone. Confirm with \`Test-Path\` before running.

### Train: 100 episodes, rollout batch of 16 transitions

\`\`\`powershell
python train_direct_p.py \`
  --mode train --train-component both \`
  --episodes 100 --seconds 120 --rollout-steps 16 \`
  --actor-lr 0.001 --critic-lr 0.001 --seed 0 \`
  --eval-every 10 --plot-every 10 \`
  --train-xref .\reference_split\train_xref.csv \`
  --output direct_results\train_direct_p_100_new
\`\`\`

### Evaluate on the **different** user-provided TEST track

\`\`\`powershell
python eval_direct_p_heldout.py \`
  --checkpoint direct_results\train_direct_p_100_new\checkpoint.npz \`
  --track-csv .\reference_split\test_xref.csv \`
  --training-reference-manifest direct_results\train_direct_p_100_new\training_reference_manifest.json \`
  --seconds 150 --output direct_results\eval_direct_p_100_new
\`\`\`

### Regenerate old experiment plots without retraining

\`\`\`powershell
python plot_direct_p_logs.py \`
  --train-dir .\direct_results\train_direct_p_ep100_roll16 \`
  --eval-dir .\direct_results\eval_direct_p_ep100_roll16 \`
  --plot-every 10
\`\`\`

These commands **do not** select the best checkpoint automatically. The
trained \`checkpoint.npz\` is the last checkpoint; do not select models using
the held-out test track.

## Cleanup notes

- A backup branch at \`backup/direct-p-before-cleanup-20261020\` preserves all
  old tracked snapshots and checkpoints. See GitHub if historical NN-Critic
  results are needed.
- \`.gitignore\` prevents new local output artifacts from being committed.
  Generated \`direct_results\` and \`reference_split\` remain on your computer.
- \`git pull\` modifies tracked files in the clone. If you modified old
  Git-tracked results locally, **back them up first and do not use
  \`git reset --hard\` or \`git clean -fd\`**.
