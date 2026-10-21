"""Run deterministic frozen checkpoints to evaluate REAL training progression.

Runs the existing frozen Actor/Critic NMPC evaluator for user-selected
1-based episode numbers, comparing TRAIN-only untrained vs learned vs
three fixed NMPC controllers on the EXACT same checked TRAIN reference.

Usage from nmpc_casadi after 100-episode training:
  python evaluate_training_curve.py \
    --train-root results_rl/gamma090_train100_seed0 \
    --reference reference_split/train_xref.csv \
    --screen-root results_screen/full_20pct \
    --output results_rl/gamma090_eval100_seed0 \
    --epochs 1 10 25 50 100 \
    --max-steps 1150 --all-train-conditions

No parameter updates or TEST. Replays deterministic policy at each
selected checkpoint. The first eval takes extra work to replay frozen
untrained Actor and fixed NMPC comparators; all comparisons use Eq.4
fixed reward and identical initial conditions. Epoch-index = user
episode number - 1 (first completed episode = 1).

The frozen evaluator performs one full NMPC simulation PER checkpoint
PER condition. It can take time on Windows; do not assume success
until all subprocesses terminate without exception and summary exists.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from screen_training_weights import verify_train_reference
from training_initialization import TRAIN_REFERENCE_SHA256


def validate_history(train_root, epochs):
    """Reject missing/interrupted, mismatched or non-gamma90 training runs."""
    root=Path(train_root).expanduser().resolve(strict=True)
    config=json.loads((root/"run_config.json").read_text(encoding="utf-8"))
    if (config.get("reference_role")!="TRAIN_ONLY" or
        config.get("train_sha256")!=TRAIN_REFERENCE_SHA256 or
        config.get("gamma")!=0.9 or
        config.get("reward_audit_verified") is not True):
        raise ValueError("Require gamma=.90 pinned TRAIN and audited fixed reward")
    history=pd.read_csv(root/"training_history.csv")
    needed={"episode","steps","mean_reward","rms_td","solver_success_pct",
            "reached_end","terminated","truncated"}
    if history.empty or not needed.issubset(history.columns):
        raise ValueError("Missing required complete training episode history")
    indices=history["episode"].to_numpy()
    if not np.array_equal(indices,np.arange(len(indices))):
        raise ValueError("Nonconsecutive or duplicated training episode indices")
    for epoch in epochs:
        if not isinstance(epoch,int) or epoch<1 or epoch>len(history):
            raise ValueError(f"Requested episode {epoch} not available; completed {len(history)}")
        if not (root/f"checkpoint_ep{epoch-1:04d}.npz").is_file():
            raise ValueError(f"Missing checkpoint for completed episode {epoch}")
    if (not np.all(np.isfinite(history[["mean_reward","rms_td","solver_success_pct"]]))
        or (history["steps"]<1).any()):
        raise ValueError("Nonfinite or empty training metrics")
    return root,config,history


def evaluation_command(*,reference,screen_root,train_root,output,epoch,
                       max_steps,all_train_conditions):
    """Canonical command delegated to already tested frozen evaluator."""
    args=[
        sys.executable,
        str(Path(__file__).with_name("evaluate_frozen_actor_critic.py")),
        "--reference",str(Path(reference).resolve()),
        "--screen-root",str(Path(screen_root).resolve()),
        "--screen-manifest",str((Path(screen_root)/"manifest.json").resolve()),
        "--checkpoint",str((Path(train_root)/f"checkpoint_ep{epoch-1:04d}.npz").resolve()),
        "--output",str(Path(output).resolve()),
        "--max-steps",str(max_steps)
    ]
    if all_train_conditions:
        args.append("--all-train-conditions")
    return args


def summarise_history(history, epochs):
    """Numerical history diagnostics; this is NOT frozen performance."""
    first=history.iloc[:min(10,len(history))]
    last=history.iloc[-min(10,len(history)):]
    return {
        "episodes_completed":len(history),
        "train_mean_reward_first10":float(first["mean_reward"].mean()),
        "train_mean_reward_last10":float(last["mean_reward"].mean()),
        "train_mean_reward_delta_last_minus_first":float(
            last["mean_reward"].mean()-first["mean_reward"].mean()),
        "train_td_rms_first10":float(first["rms_td"].mean()),
        "train_td_rms_last10":float(last["rms_td"].mean()),
        "train_mean_solver_success_pct":float(history["solver_success_pct"].mean()),
        "train_episode_reached_end_count":int(history["reached_end"].astype(bool).sum()),
        "train_episode_truncated_count":int(history["truncated"].astype(bool).sum()),
        "requested_frozen_checkpoints":list(epochs),
        "caveat":"Stochastic TRAIN reward alone is NOT a trained-policy performance or held-out TEST metric",
    }


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-root",type=Path,required=True)
    ap.add_argument("--reference",type=Path,required=True)
    ap.add_argument("--screen-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--epochs",type=int,nargs="+",required=True,
                    help="ONE-BASED completed episodes e.g. 1 10 25 50 100")
    ap.add_argument("--max-steps",type=int,required=True)
    ap.add_argument("--all-train-conditions",action="store_true")
    args=ap.parse_args(argv)
    if args.max_steps<1 or any(ep<1 for ep in args.epochs) or (
        len(args.epochs)!=len(set(args.epochs)) or
        sorted(args.epochs)!=args.epochs):
        ap.error("Positive max steps and unique ascending one-based epochs required")
    ref,digest=verify_train_reference(args.reference)
    root,config,history=validate_history(args.train_root,args.epochs)
    screen_root=Path(args.screen_root).expanduser().resolve(strict=True)
    manifest=json.loads((screen_root/"manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("train_sha256")!=digest or
        manifest.get("profile")!="full" or
        manifest.get("empirical_all_pass") is not True or
        manifest.get("pass_samples")!=27):
        ap.error("Require 27 PASS full screening on SAME pinned TRAIN reference")
    out=args.output.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        ap.error("Output nonempty: do not overwrite existing frozen evaluation")
    out.mkdir(parents=True,exist_ok=True)

    records=[]
    expected_controllers={
        "trained_deterministic","untrained_deterministic",
        "learning_initial","strong_fixed_baseline",
        "stage_lower_terminal_upper"
    }
    for epoch in args.epochs:
        subdir=out/f"epoch_{epoch:04d}"
        command=evaluation_command(
            reference=ref,screen_root=screen_root,train_root=root,
            output=subdir,epoch=epoch,max_steps=args.max_steps,
            all_train_conditions=args.all_train_conditions)
        print(f"Evaluating frozen episode {epoch} / {args.epochs[-1]}",flush=True)
        # Make output visible in real time; reject any failure instead of
        # silently dropping a checkpoint or claiming false improvement.
        subprocess.run(command,check=True,cwd=Path(__file__).resolve().parent)
        table=pd.read_csv(subdir/"comparison.csv")
        if set(table["controller"])!=expected_controllers:
            raise ValueError("Missing trained/untrained/fixed comparison")
        table["checkpoint_episode_1based"]=epoch
        records.append(table)
        combined=pd.concat(records,ignore_index=True)
        combined.to_csv(out/"learning_curve_comparison.csv",index=False)
    summary=summarise_history(history,args.epochs)
    summary.update(
        type="TRAIN_ONLY_MULTI_EPISODE_FROZEN_LEARNING_CURVE",
        gamma=.90,train_sha256=digest,
        evaluated_rows=len(combined),
        all_train_conditions=bool(args.all_train_conditions),
        epochs=list(args.epochs),
        no_parameter_updates=True,
        no_test_used=True,
        no_claim_of_stability=True,
        reminder="Compare frozen trained vs frozen untrained on SAME initial condition and SAME reward; fixed baselines are SCREENING log reuses.",
    )
    (out/"learning_curve_summary.json").write_text(
        json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({
        "status":"FROZEN_TRAIN_LEARNING_CURVE_COMPLETE",
        "episodes_evaluated":args.epochs,
        "n_rows":len(combined),
        "summary_file":str(out/"learning_curve_summary.json"),
        "comparison_file":str(out/"learning_curve_comparison.csv"),
        "heldout_test_used":False,
    }),flush=True)


if __name__=="__main__":
    main()
