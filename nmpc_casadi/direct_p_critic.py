"""Exactly six directly learned Critic parameters for paper RL-NMPC.

Authoritative choices:
  Paper MPC_RL.pdf Remark 4 allows a parameterized (non-NN) Critic.
  Eq. (6)-(7) uses one-step TD error and 0.5*squared TD loss.
  Eq. (29)-(33) requires positive weights in an admissible projected domain.

User-approved specialization:
  P=diag(p[0],...,p[5]), V_P(s)=-e(s)^T P e(s), Qf=P.

There is NO neural Critic, extra value baseline, parameter multiplier network,
multi-step target, gradient normalization/clipping, or hidden reward shaping.
The caller provides positive physically meaningful bounds, an initial P,
gamma and the learning rate; they are NOT silently chosen here.

This semi-gradient TD(0) treats the bootstrap target as constant during a
single update. The six p_i are in PHYSICAL Qf units, not normalized features.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from rl_paper_core import critic_squared_td_loss, project_box_admissible, td_error


STATE_NAMES = ("vx", "vy", "psi", "r", "X", "Y")


def error_from_state(x, xref):
    """Full 6-state reference error, including the same wrapped heading as NMPC."""
    x = np.asarray(x, dtype=float)
    xref = np.asarray(xref, dtype=float)
    if x.shape != (6,) or xref.shape != (6,) or (
        not np.all(np.isfinite(x)) or not np.all(np.isfinite(xref))
    ):
        raise ValueError("x and xref must each be finite six-state vectors")
    err = x - xref
    err[2] = np.arctan2(np.sin(err[2]), np.cos(err[2]))
    return err


def _six_finite(x, name):
    v = np.asarray(x, dtype=float)
    if v.shape != (6,) or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must have six finite elements")
    return v.copy()


@dataclass(frozen=True)
class CriticStep:
    """One paper TD(0) transition; all diagnostics from BEFORE the update."""
    td_error: float
    reward: float
    value_current: float
    value_next: float
    target: float
    loss: float
    semi_gradient: tuple[float, ...]
    p_before: tuple[float, ...]
    p_after: tuple[float, ...]
    terminated: bool


class SixParameterTDCritic:
    """A single directly trained six-dimensional vector, no NN parameters.

    The nominal/bound selection is explicitly owned by the caller.
    Pass lower and upper bounds from an externally approved admissible set.
    These box bounds alone do not prove the terminal-set stability conditions
    in the paper's Theorem 2.
    """

    def __init__(self, initial_p, lower_p, upper_p):
        initial = _six_finite(initial_p, "initial_p")
        lower = _six_finite(lower_p, "lower_p")
        upper = _six_finite(upper_p, "upper_p")
        if np.any(lower <= 0) or np.any(upper <= lower):
            raise ValueError("require 0 < lower_p < upper_p componentwise")
        if np.any(initial < lower) or np.any(initial > upper):
            raise ValueError("initial P must lie inside the supplied bounds")
        self._p = initial
        self.lower = lower
        self.upper = upper

    @property
    def p(self):
        """Defensive copy: direct learning is exclusively through TD updates."""
        return self._p.copy()

    def terminal_diagonal(self):
        return self.p

    def terminal_matrix(self):
        return np.diag(self._p)

    def value(self, error):
        e = _six_finite(error, "error")
        return -float(np.dot(self._p, e**2))

    def td_step(self, error, next_error, reward, gamma, learning_rate, *, terminated=False):
        """Perform exactly one projected TD(0) semi-gradient step.

        learning_rate may be a scalar, or exactly six positive component
        rates for [vx,vy,psi,r,X,Y]. This is an explicit diagonal SGD
        preconditioner, not an extra neural Critic or reward adjustment.

        For terminal=true, the future value is zero. A time-limit truncation
        must be passed as terminated=false so it continues to bootstrap.
        No derivative is taken through next_error or the next-state value.
        """
        e = _six_finite(error, "error")
        en = _six_finite(next_error, "next_error")
        if isinstance(terminated, (bool, np.bool_)):
            terminated = bool(terminated)
        else:
            raise ValueError("terminated must be boolean, not a truncation flag")
        # Allow a common scalar LR (the paper-aligned baseline) or six
        # independently selected physical-P rates for an explicit ablation.
        rate=np.asarray(learning_rate,dtype=float)
        if (rate.shape not in ((),(6,)) or
            not np.all(np.isfinite(rate)) or np.any(rate<=0)):
            raise ValueError(
                "reward/learning rate: require finite positive scalar or six LR values")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        before = self._p.copy()
        value_now = -float(np.dot(before, e**2))
        value_next = 0.0 if terminated else -float(np.dot(before, en**2))
        d = float(td_error(float(reward), value_now, value_next, gamma))
        loss = float(critic_squared_td_loss(d))
        # dL/dp_i = (V(s)-target) * dV/dp_i = delta * e_i**2
        # because V=-sum(p_i*e_i**2), delta=target-V.
        semi_grad = d * e**2
        new_p = project_box_admissible(before - rate*semi_grad,
                                      self.lower, self.upper)
        self._p = new_p.copy()
        return CriticStep(
            td_error=d, reward=float(reward), value_current=value_now,
            value_next=value_next,
            target=float(reward)+float(gamma)*value_next,
            loss=loss,
            semi_gradient=tuple(float(x) for x in semi_grad),
            p_before=tuple(float(x) for x in before),
            p_after=tuple(float(x) for x in self._p),
            terminated=terminated,
        )
