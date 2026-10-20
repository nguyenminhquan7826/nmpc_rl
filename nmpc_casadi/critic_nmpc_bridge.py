"""Pass Actor Q/R and directly learned Critic P into one NMPC solve.

The ONLY Critic->NMPC relationship is Qf=diag(P).
The NN Actor is not constructed in this module. Stage diagonals must be
supplied by that Actor and screened against the paper's admissible domain.
"""
from __future__ import annotations

import numpy as np
from direct_p_critic import SixParameterTDCritic
from nmpc_casadi import pack_parameters


def actor_critic_nmpc_parameters(x, u_prev, xref_horizon, p_model,
                                 actor_q_diagonal, actor_r_diagonal,
                                 critic: SixParameterTDCritic):
    """Pack parameters for build_nmpc_solver(..., adaptive_cost_weights=True)."""
    if not isinstance(critic, SixParameterTDCritic):
        raise TypeError("critic must be SixParameterTDCritic")
    q = np.asarray(actor_q_diagonal, dtype=float)
    r = np.asarray(actor_r_diagonal, dtype=float)
    if q.shape != (6,) or r.shape != (2,) or not all(
        np.all(np.isfinite(y)) and np.all(y > 0) for y in (q, r)
    ):
        raise ValueError("Actor must supply positive finite Q[6], R[2] diagonals")
    return pack_parameters(
        x, u_prev, xref_horizon, p_model,
        weights=(q, r, critic.terminal_diagonal()),
    )
