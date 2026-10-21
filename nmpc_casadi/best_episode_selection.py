"""Select best completed TRAIN episode and best deterministic frozen replay.

Two definitions must never be conflated: stochastic training tracking
and deterministic post-training closed-loop evaluation. TEST is not used.
"""
from __future__ import annotations
import math


def select_best_train_episode(records, min_solver_success_pct=99.0):
    """Best feasible stochastic episode by mean TRAIN CTE, then reward.

    Returns None when no TRAIN episode reached the end with sufficiently
    reliable NMPC solves. The 'episode' field is zero-based.
    """
    if not 0.0 <= float(min_solver_success_pct) <= 100.0:
        raise ValueError("Invalid solver-success threshold")
    feasible=[]
    for row in records:
        if (bool(row["reached_end"]) and
            float(row["solver_success_pct"])>=min_solver_success_pct and
            math.isfinite(float(row["mean_cte_m"])) and
            math.isfinite(float(row["mean_reward"]))):
            feasible.append(row)
    if not feasible:
        return None
    return min(feasible,key=lambda r: (
        float(r["mean_cte_m"]),-float(r["mean_reward"]),int(r["episode"])))


def select_best_frozen_checkpoint(records, min_solver_success_pct=99.0):
    """Best deterministic completed trained checkpoint, using post-capture CTE.

    Accepts dictionary records from comparison.csv; skips fixed NMPC and
    incomplete/truncated/failure-terminal replays. This only compares
    checkpoints ACTUALLY evaluated, not all past episodes.
    """
    if not 0.0 <= float(min_solver_success_pct) <= 100.0:
        raise ValueError("Invalid solver-success threshold")
    candidates=[]
    for row in records:
        name=str(row["controller"])
        if not name.startswith("trained_episode_"):
            continue
        if (bool(row["reached_end"]) and
            not bool(row["failure_terminal"]) and
            not bool(row["truncated"]) and
            float(row["solver_success_pct"])>=min_solver_success_pct and
            math.isfinite(float(row["mean_cte_post_m"])) and
            math.isfinite(float(row["mean_reward"]))):
            candidates.append(row)
    if not candidates:
        return None
    return min(candidates,key=lambda r: (
        float(r["mean_cte_post_m"]),-float(r["mean_reward"]),
        int(str(r["controller"]).rsplit("_",1)[1])))
