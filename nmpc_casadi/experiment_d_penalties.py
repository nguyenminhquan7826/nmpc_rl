"""Experiment D: optional TRAIN-only solver and catastrophic-termination penalties.

Never modify the fixed Eq.(4) task reward, TEST, NMPC dynamics or frozen
evaluation. Unrecoverable failures mean BOTH normal and fallback solves
failed (step_diagnostics['solver_success'] == False).
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RewardBreakdown:
    paper_reward: float
    solver_penalty: float
    terminal_penalty: float
    training_reward: float


def compute_training_reward(paper_reward, diagnostic, *,
                            solver_failure_penalty=0.0,
                            failure_terminal_penalty=0.0):
    """Return paper reward and nonnegative costs for TRAIN only.

    Failure-terminal cost includes invalid plant state and the third
    consecutive failed solve; a normal reached_end is NOT penalized.
    This changes the RL objective; it is not the original paper Eq.(4).
    """
    for name, value in (
        ("paper_reward", paper_reward),
        ("solver_failure_penalty", solver_failure_penalty),
        ("failure_terminal_penalty", failure_terminal_penalty),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if solver_failure_penalty < 0 or failure_terminal_penalty < 0:
        raise ValueError("Safety penalty coefficients must be nonnegative")
    if not isinstance(diagnostic, dict):
        raise TypeError("Expected one NMPC step diagnostics dictionary")
    for name in ("solver_success", "failure_terminal", "reached_end"):
        if name not in diagnostic or not isinstance(diagnostic[name], bool):
            raise ValueError(f"Missing boolean NMPC diagnostic {name}")
    if diagnostic["failure_terminal"] and diagnostic["reached_end"]:
        raise ValueError("Episode cannot be both failure-terminal and successfully complete")
    solver_cost = float(solver_failure_penalty) * (not diagnostic["solver_success"])
    terminal_cost = float(failure_terminal_penalty) * diagnostic["failure_terminal"]
    train_reward = float(paper_reward) - solver_cost - terminal_cost
    return RewardBreakdown(float(paper_reward),float(solver_cost),
                           float(terminal_cost),float(train_reward))
