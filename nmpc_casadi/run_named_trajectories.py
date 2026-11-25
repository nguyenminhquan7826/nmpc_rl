"""Run an isolated RL-NMPC experiment on the selected 8m routes.

Example (from nmpc_casadi):
    python run_named_trajectories.py --train spiral_8m --evaluate zigzag_8m --episodes 30
    python run_named_trajectories.py --train zigzag_8m --evaluate spiral_8m --episodes 30

The evaluation checkpoint is selected exclusively from TRAIN, never the held-out path.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
os.environ["NMPC_RL_MINIMAL_LAYOUT"]="1"

import experiment_config as cfg
from trajectory_selection import NEW_TRACKS
from train_actor_critic import main as train_main
from evaluate_trajectory_frozen import main as frozen_main

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", choices=NEW_TRACKS, required=True)
    ap.add_argument("--evaluate", choices=NEW_TRACKS, required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--steps", type=int, default=1250,
                    help="Ts=0.071s; 8m at 0.1m/s needs about 1127 samples")
    ap.add_argument("--eval-steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Evaluate an existing checkpoint; skip training")
    ap.add_argument("--online-eval", action="store_true",
                    help="Evaluate using online adaptation instead of frozen generalization")
    args=ap.parse_args(argv)
    if args.episodes<1 or args.steps<1 or args.eval_steps<1:
        ap.error("episodes and steps must be positive")
    if args.online_eval and args.checkpoint is None:
        # Works with the checkpoint produced by the immediately preceding training.
        pass

    out = (args.output if args.output is not None else
           ROOT/"results_trajectories"/f"{args.train}_to_{args.evaluate}_seed{args.seed}")
    out = out.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        ap.error("Output directory must be empty; prior results are preserved")

    train_out=out/"training"
    eval_out=out/"evaluation"
    if args.checkpoint is None:
        argv_train=[
            "--trajectory",args.train,
            "--use-internal-box","--box-margin",str(cfg.WEIGHT_BOX_MARGIN),
            "--output",str(train_out),
            "--acknowledge-empirical-only",
            "--reward-candidate","legacy",
            "--gamma",str(cfg.GAMMA),
            "--sigma-lower",str(cfg.SIGMA_MIN),
            "--sigma-initial",str(cfg.SIGMA_INIT),
            "--sigma-upper",str(cfg.SIGMA_MAX),
            "--actor-lr",str(cfg.ACTOR_LR),
            "--critic-lr",str(cfg.CRITIC_LR),
            "--critic-update-mode",str(cfg.CRITIC_UPDATE_MODE),
            "--solver-failure-penalty",str(cfg.SOLVER_FAILURE_PENALTY),
            "--failure-terminal-penalty",str(cfg.FAILURE_TERMINAL_PENALTY),
            "--actor-mu-reg-coef",str(cfg.ACTOR_MU_REG_COEF),
            "--actor-mu-reg-threshold",str(cfg.ACTOR_MU_REG_THRESHOLD),
            "--actor-output-scales",*[str(v) for v in cfg.ACTOR_OUTPUT_LR_FACTORS],
            "--critic-component-scales",*[str(v) for v in cfg.CRITIC_P_LR_FACTORS],
            "--episodes",str(args.episodes),
            "--steps-per-episode",str(args.steps),
            "--seed",str(args.seed),
            "--ey",str(cfg.INITIAL_EY_M),
            "--heading-deg",str(cfg.INITIAL_HEADING_DEG),
        ]
        train_main(argv_train)
        checkpoint=train_out/"checkpoint_best_train.npz"
        if not checkpoint.is_file():
            checkpoint=train_out/f"checkpoint_ep{args.episodes-1:04d}.npz"
        if not checkpoint.is_file():
            raise RuntimeError("TRAIN did not produce a checkpoint")
    else:
        checkpoint=args.checkpoint.expanduser().resolve(strict=True)
    if args.online_eval:
        from evaluate_controller_modes import main as online_main
        online_main([
            "--trajectory",args.evaluate,"--checkpoint",str(checkpoint),
            "--output",str(eval_out),"--max-steps",str(args.eval_steps),
            "--seeds",str(args.seed),"--save-data",
        ])
    else:
        frozen_main([
            "--trajectory",args.evaluate,"--checkpoint",str(checkpoint),
            "--output",str(eval_out),"--max-steps",str(args.eval_steps),
        ])
    print(f"TRAIN: {args.train} | EVAL: {args.evaluate}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Evaluation: {eval_out/'comparison.csv'}")
    return out


if __name__=="__main__":
    main()
