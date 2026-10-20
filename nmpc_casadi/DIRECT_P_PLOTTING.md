# Direct-P PNG logging: train and held-out evaluation

This is **plotting only**. It does not change the Actor NN, six-parameter
Critic P, one-step TD(0) target, or NMPC solver.

## How to update your local clone

From PowerShell:


    cd D:\GitHub\RL_NMPC\Code\nmpc_rl_direct_p
    git pull --ff-only origin experiment/direct-quadratic-td-critic
    cd nmpc_casadi

## Auto-generated plots for NEW runs

Train 100 episodes, 16-transition batches:


    python train_direct_p.py `
      --mode train `
      --train-component both `
      --episodes 100 --seconds 120 --rollout-steps 16 `
      --actor-lr 0.001 --critic-lr 0.001 --seed 0 `
      --eval-every 10 --plot-every 10 `
      --train-xref .\reference_split\train_xref.csv `
      --output direct_results\train_direct_p_ep100_roll16_NEW

Test an already trained model on the independent 10.70 m user-supplied CSV:


    python eval_direct_p_heldout.py `
      --checkpoint direct_results\train_direct_p_ep100_roll16_NEW\checkpoint.npz `
      --track-csv .\reference_split\test_xref.csv `
      --training-reference-manifest direct_results\train_direct_p_ep100_roll16_NEW\training_reference_manifest.json `
      --seconds 150 --seed 0 `
      --output direct_results\eval_direct_p_ep100_roll16_NEW

By default plotting is on. Pass `--no-plots` to train or evaluate if you only want
the CSVs. `--plot-every 10` plots episode 0, every tenth episode, and
the last episode; all other per-step logs remain as CSVs.

## Recreate images from EXISTING results — NO retraining

**Use this for your previous 100-episode training and test.** It does not
overwrite the model, episodes, evaluation metrics or CSVs:


    python plot_direct_p_logs.py `
      --train-dir .\direct_results\train_direct_p_ep100_roll16 `
      --eval-dir .\direct_results\eval_direct_p_ep100_roll16 `
      --plot-every 10

## Output file names

Training root:
- `training_return.png`
- `training_mean_cte_post.png`
- `training_heading_post.png`
- `training_solver_success.png`
- `training_terminal_weights.png` (physical P and normalized multipliers)

Detailed training data (episode 0, 9, 19, ..., 99 by default):
- `plots/episode_0099_trajectory.png`
- `plots/episode_0099_cte_response.png`
- `plots/episode_0099_weights_qrqf.png` (Q6/R2/Qf6 vs simulation time)
- `plots/episode_0099_td_losses.png` (TD RMS, Critic and Actor loss)
- `plots/validation_ep0099_trajectory.png` and related Q/R/Qf plots for
  **new runs**; historical validation logs are retained as CSV.

Held-out evaluation root:
- `tracking_metric_comparison.png`
- `timing_fixed_vs_ac.png`
- `nominal_trajectory.png`, `payload_trajectory.png`,
  `tire_segment_trajectory.png`, `rolling_segment_trajectory.png`
- `{profile}_fixed_vs_ac_response.png` and `{profile}_fixed_vs_ac.png`
- `{profile}_{controller}_weights_qrqf.png` for **every** test controller
- `{profile}_{controller}_cte_response.png`
- `{profile}_online_td_losses.png`

The **full evaluation reference** is drawn from the staged evaluation XREF when
available; plot curves use the actual CSV records, never synthesized metrics.
The test path is never used for training or checkpoint selection.
