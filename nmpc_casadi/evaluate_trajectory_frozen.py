"""Frozen, held-out evaluation on either checksum-pinned new 8 m reference.

No learning, stochastic action sampling, or test-driven checkpoint selection.
Unlike evaluate_controller_modes.py, this compares the ACTUAL learned policy
without resetting output heads or changing checkpoint Q/R/P bounds.
"""
from __future__ import annotations
import argparse
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

os.environ["NMPC_RL_MINIMAL_LAYOUT"] = "1"
import baseline as b
from cad_eval_reference import prepare_cad_evaluation_reference, sha256_file
from direct_p_critic import SixParameterTDCritic
from evaluate_frozen_actor_critic import frozen_rollout, load_frozen_agents
from training_initialization import GOOD_NMPC, LEARNING_INITIAL
from train_rl_environment import TrainEnvironment
from trajectory_selection import TRACK_CHOICES, identify_checkpoint_train, resolve_named_track


class FixedActor:
    def __init__(self, q, r):
        self.q = np.asarray(q, float)
        self.r = np.asarray(r, float)
        self.params = {}

    def propose(self, observation, *, deterministic=False):
        if not deterministic:
            raise ValueError("Fixed controller is deterministic")
        return SimpleNamespace(
            q_diagonal=self.q.copy(),
            r_diagonal=self.r.copy(),
            sigma=np.zeros(8),
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--trajectory", choices=TRACK_CHOICES)
    group.add_argument("--reference", type=Path, help="Explicit held-out CSV")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--ey", type=float, default=0.05)
    ap.add_argument("--heading-deg", type=float, default=0.0)
    args = ap.parse_args(argv)
    if args.max_steps < 1:
        ap.error("--max-steps must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        ap.error("Output folder must be empty; previous results are protected")
    with np.load(args.checkpoint, allow_pickle=False) as data:
        config = json.loads(str(data["metadata"].item()))
    train_name, train_hash = identify_checkpoint_train(config)
    lo = tuple(np.asarray(config[f"{key}_lower"],float) for key in ("Q","R","P"))
    hi = tuple(np.asarray(config[f"{key}_upper"],float) for key in ("Q","R","P"))
    (initial_actor, initial_critic), (trained_actor, trained_critic), config = load_frozen_agents(
        args.checkpoint, lo, hi, expected_gamma=float(config["gamma"])
    )
    if args.trajectory in ("spiral_8m", "zigzag_8m"):
        reference, _ = resolve_named_track(args.trajectory)
        eval_name = args.trajectory
    elif args.trajectory == "pinned":
        from experiment_config import TRAIN_CSV
        from screen_training_weights import verify_train_reference
        reference, _ = verify_train_reference(TRAIN_CSV)
        eval_name = "pinned"
    else:
        reference = args.reference.expanduser().resolve(strict=True)
        eval_name = "external"
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nmpc-frozen-ref-") as tmp:
        details, staged = prepare_cad_evaluation_reference(
            reference, Path(tmp), b.TAIL_LENGTH_M,
            min_speed_mps=float(b.cfg["vx_min"]),
            max_speed_mps=min(float(b.TEST_SPEED_MPS),float(b.cfg["vx_max"])),
            wheelbase_m=float(b.WHEELBASE_M),
            steering_limit_rad=min(abs(float(b.cfg["u_min"][1])),
                                   abs(float(b.cfg["u_max"][1]))),
        )
        b.REF_INFO = details
        b.TAIL_FILE = Path(staged["tail_file"])
        b.CAD_S_END = float(details["cad_s_end"])
        q0, r0, p0 = LEARNING_INITIAL.arrays()
        good_q, good_r, good_p = GOOD_NMPC.arrays()
        comparisons = [
            ("strong_fixed_nmpc", FixedActor(good_q,good_r),
             SixParameterTDCritic(good_p,lo[2],hi[2])),
            ("untrained_actor_critic", initial_actor,initial_critic),
            ("trained_frozen_actor_critic", trained_actor,trained_critic),
        ]
        records = []
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7,6))
        ax.plot(details["cad_X"], details["cad_Y"],"k--",label="8m reference")
        for label,actor,critic in comparisons:
            env = TrainEnvironment(
                train_length_m=b.CAD_S_END, q_initial=q0,
                r_initial=r0, p_initial=p0,
                task_qe=np.asarray(config["Qe_fixed"],float),
                task_re=np.asarray(config["Re_fixed"],float),
                max_steps=args.max_steps, ey=args.ey, heading_deg=args.heading_deg,
            )
            summary,log = frozen_rollout(env,actor,critic,float(config["gamma"]),label)
            summary["train_trajectory"] = train_name
            summary["evaluation_trajectory"] = eval_name
            summary["same_reference_as_train"] = bool(staged["reference_sha256"] == train_hash)
            records.append(summary)
            log.to_csv(args.output/f"{label}.csv",index=False)
            ax.plot(log["X_m"],log["Y_m"],label=label)
            print(json.dumps(summary),flush=True)
        ax.set_aspect("equal",adjustable="datalim")
        ax.set(xlabel="X (m)",ylabel="Y (m)",title=f"Frozen evaluation: {train_name} -> {eval_name}")
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output/"trajectory_comparison.png",dpi=170)
        plt.close(fig)
        pd.DataFrame(records).to_csv(args.output/"comparison.csv",index=False)
        manifest = {
            "type":"FROZEN_CROSS_TRAJECTORY_EVALUATION",
            "no_learning_or_sampling":True,
            "train_trajectory":train_name,
            "evaluation_trajectory":eval_name,
            "train_reference_sha256":train_hash,
            "evaluation_reference_sha256":staged["reference_sha256"],
            "same_reference_as_train":staged["reference_sha256"]==train_hash,
            "checkpoint_sha256":sha256_file(args.checkpoint),
            "original_checkpoint_bounds_preserved":True,
            "not_a_stability_certificate":True,
            "max_steps":args.max_steps,
        }
        (args.output/"evaluation_config.json").write_text(
            json.dumps(manifest,indent=2),encoding="utf-8"
        )
    return records


if __name__ == "__main__":
    main()
