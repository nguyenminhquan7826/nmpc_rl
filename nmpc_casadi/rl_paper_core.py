"""Paper equations shared by an eventual Actor-Critic-NMPC implementation.

IMPORTANT
---------
This module implements only SOURCE-DEFINED, architecture-independent
mathematics. It does not choose an Actor network, a Critic architecture,
an advantage normalizer, or a learning schedule. No backpropagation
passes through NMPC, and the frozen NMPC solver is not changed here.

Sources:
  MPC_RL.pdf: Eq. (4), (6), (7), (8), (29)-(33).
  NMPC(1).docx: Chapter II, Actor gradient derivation.

All state errors follow [vx, vy, psi, r, X, Y], and input increments
follow [a, delta]. Cost/Reward weights Qe,Re refer to the task reward;
they are NOT necessarily the state/input weights adapted for NMPC.
"""

from __future__ import annotations

import numpy as np


def _positive_diagonal(diagonal, size: int, name: str) -> np.ndarray:
    a = np.asarray(diagonal, dtype=float)
    if a.shape != (size,) or not np.all(np.isfinite(a)) or not np.all(a > 0.):
        raise ValueError(f"{name} must have {size} finite strictly positive entries")
    return a


def state_and_input_increment_cost(
    error, control_increment, task_q_diagonal, task_r_diagonal,
):
    """Paper Eq. (4) cost BEFORE negation: e.T Qe e + du.T Re du.

    The paper reward Eq. (4) has no global 1/2 factor. The NMPC
    optimization in Algorithm 1 uses a 1/2 convention instead.
    """
    q = _positive_diagonal(task_q_diagonal, 6, "task Qe")
    r = _positive_diagonal(task_r_diagonal, 2, "task Re")
    e = np.asarray(error, dtype=float)
    du = np.asarray(control_increment, dtype=float)
    if e.shape[-1:] != (6,) or du.shape[-1:] != (2,):
        raise ValueError("error must end in 6 and control increment in 2")
    if e.shape[:-1] != du.shape[:-1]:
        raise ValueError("error/control increment must have matching leading dimensions")
    if not np.all(np.isfinite(e)) or not np.all(np.isfinite(du)):
        raise ValueError("error/control increment must be finite")
    return np.sum(e**2 * q, axis=-1) + np.sum(du**2 * r, axis=-1)


def paper_reward(error, control_increment, task_q_diagonal, task_r_diagonal):
    """Eq. (4): r_k = -(e_k.T Qe e_k + delta_u_k.T Re delta_u_k)."""
    return -state_and_input_increment_cost(
        error, control_increment, task_q_diagonal, task_r_diagonal
    )


def td_error(reward, value_current, value_next, gamma):
    """Exact Eq. (6): delta = r + gamma * V(s_next) - V(s).

    The paper prints a continuing-task formula, with no termination mask.
    For episodic training, a caller must explicitly supply value_next=0
    at a genuinely terminal transition. Do not silently treat time
    truncation as termination.
    """
    if not np.isfinite(gamma) or not 0 < gamma <= 1.:
        raise ValueError("gamma must be finite and in (0, 1]")
    a = [np.asarray(x, dtype=float) for x in (reward, value_current, value_next)]
    if not all(np.all(np.isfinite(x)) for x in a):
        raise ValueError("reward/current value/next value must be finite")
    return a[0] + gamma * a[2] - a[1]


def critic_squared_td_loss(delta):
    """Eq. (7): 0.5 * delta**2; no unapproved n-step/Huber term."""
    d = np.asarray(delta, dtype=float)
    if not np.all(np.isfinite(d)):
        raise ValueError("TD residual must be finite")
    return 0.5 * d**2


def actor_log_policy_objective(log_probability, frozen_td):
    """Descent loss corresponding to policy-gradient ASCENT of Eq. (8).

    Frozen TD is the source-provided advantage approximation. Differentiation
    must flow only through log_probability, not through frozen_td.
    This is a scalar reference formula, NOT a trainable NN implementation.
    """
    logp = np.asarray(log_probability, dtype=float)
    delta = np.asarray(frozen_td, dtype=float)
    if not np.all(np.isfinite(logp)) or not np.all(np.isfinite(delta)):
        raise ValueError("log probability and TD error must be finite")
    if logp.shape != delta.shape:
        raise ValueError("log probability and TD error must have equal shapes")
    if delta.size == 0:
        raise ValueError("At least one sample is required")
    return float(-np.mean(delta * logp))


def project_box_admissible(candidate, lower, upper):
    """Euclidean projection onto a compact box, Eq. (29) and (31)-(33).

    This validates a *box* and positive weights. It does NOT certify that
    every point inside the box ensures NMPC stability (paper Remark 5),
    which requires an offline closed-loop feasibility/terminal-set screen.
    """
    x = np.asarray(candidate, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    if x.shape != low.shape or x.shape != high.shape or x.size == 0:
        raise ValueError("candidate, lower and upper must have matching non-empty shapes")
    if not all(np.all(np.isfinite(y)) for y in (x, low, high)):
        raise ValueError("all projection inputs must be finite")
    if np.any(low <= 0) or np.any(high <= low):
        raise ValueError("require 0 < lower < upper componentwise")
    return np.clip(x, low, high)
