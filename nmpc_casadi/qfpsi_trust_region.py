"""Evaluation-only Qf_psi(s) trust projection at the NMPC input boundary.

This diagnostic deliberately keeps the CURRENT Critic TD-Huber learning
unmodified. Only the applied terminal heading weight in the NMPC action is
projected to a trust region around a deep-copied PRETRAINED Critic output
at the very same observation.

This isolates the control-interface intervention from the previous
qfpsi_isolation.py experiment which ALSO changed the Critic gradient.
It does NOT certify closed-loop stability. No terminal set or Lyapunov
decrease proof is provided by projection alone.

The controller is a composite of the adapted checkpoint and immutable
pretrained Qf_psi(s). A plain Agent.save() would not reproduce it.
"""
from copy import deepcopy
import numpy as np
import rl_nmpc as rl


QFPSI_INDEX = 2


class QfPsiTrustProjector:
    def __init__(self, pretrained_critic, trust_pct):
        pct = float(trust_pct)
        if not np.isfinite(pct) or not (0.0 <= pct <= 1.0):
            raise ValueError('Qf_psi trust percentage must be finite in [0, 1]')
        self.trust_pct = pct
        # CRITICAL: snapshot at checkpoint load, before *any* online learning.
        self.pretrained_critic = deepcopy(pretrained_critic)

    def reference_weight(self, state):
        raw, _ = self.pretrained_critic.forward(np.asarray(state, dtype=float))
        sig = rl.sigmoid(raw[..., QFPSI_INDEX])
        return rl.F_BASE[QFPSI_INDEX] * (
            rl.FLO + (rl.FHI - rl.FLO) * sig
        )

    def project(self, weights, state):
        if len(weights) != 3:
            raise ValueError('Expected (Q,R,Qf) from Agent.propose')
        q, r, f = [np.asarray(w, dtype=float) for w in weights]
        if q.shape != (6,) or r.shape != (2,) or f.shape != (6,):
            raise ValueError('Incorrect controller weight vector dimensions')
        if any(not np.all(np.isfinite(w)) or np.any(w <= 0)
               for w in (q, r, f)):
            raise ValueError('Non-positive/nonfinite NMPC weight candidate')

        reference = float(self.reference_weight(state))
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError('Invalid pretrained Qf_psi(s) reference')
        candidate = float(f[QFPSI_INDEX])
        lower = reference * (1.0 - self.trust_pct)
        upper = reference * (1.0 + self.trust_pct)
        applied = float(np.clip(candidate, lower, upper))

        # Only the SOLVER INPUT changes. Do not modify the Agent/Critic/value,
        # TD targets, optimizer, checkpoint, policy sampling or original arrays.
        f_safe = f.copy()
        f_safe[QFPSI_INDEX] = applied
        if not (lower - 1e-9 <= f_safe[QFPSI_INDEX] <= upper + 1e-9):
            raise RuntimeError('Applied Qf_psi escaped trust interval')

        return (q.copy(), r.copy(), f_safe), {
            'qfpsi_pretrained': reference,
            'qfpsi_raw_candidate': candidate,
            'qfpsi_applied': applied,
            'qfpsi_raw_ratio': candidate / reference,
            'qfpsi_applied_ratio': applied / reference,
            'qfpsi_trust_lower': lower,
            'qfpsi_trust_upper': upper,
            'qfpsi_projected': int(not np.isclose(
                candidate, applied, rtol=1e-12, atol=1e-9,
            )),
        }
