"""Phase 2D: controlled signed sensitivity sweep of terminal Qf_psi(s).

This is a PURE evaluation intervention, not terminal-weight learning.

The input checkpoint's Critic supplies six state-dependent pretrained
terminal weights through an independently cloned, fixed terminal network.
Five paired runs multiply ONLY Qf_psi(s) by a constant signed factor:

    Qf_psi_applied(s) = (1 + signed_frac) * Qf_psi_pretrained(s)

with signed_frac = -0.05, -0.02, 0.00, +0.02, +0.05.
The other five Qf(s) entries are unchanged and Actor and Value Critic
learn identically to split Phase 2A (though their trajectories diverge).
Actor Gaussian exploration draws follow the same checkpoint seed; no
new action sampling or terminal-gradient step is introduced.

A signed perturbation around pretrained Qf is NOT a stability guarantee.
Keep the change in evaluation only; never modify selected checkpoint.
"""

import numpy as np
from split_critic_eval import SplitValueTerminalPolicy


OFFSETS = (-0.05, -0.02, 0.0, +0.02, +0.05)
PSI = 2


class SignedTerminalPsiPolicy(SplitValueTerminalPolicy):
    def __init__(self, agent, signed_frac):
        super().__init__(agent)
        frac = float(signed_frac)
        if not np.isfinite(frac) or not (-0.05 <= frac <= 0.05):
            raise ValueError('Qf_psi signed offset must be finite in [-0.05,+0.05]')
        if not any(abs(frac - x) < 1e-12 for x in OFFSETS):
            raise ValueError('Qf_psi offset must be an explicit Phase 2D sweep point')
        self.signed_frac = float(next(x for x in OFFSETS if abs(frac-x)<1e-12))
        self.audit = []

    def propose(self, s, explore):
        (q, r, f0), z = super().propose(s, explore)
        reference = float(f0[PSI])
        if not np.isfinite(reference) or reference <= 0.:
            raise ValueError('Invalid pretrained state-dependent Qf_psi')
        applied = float(reference * (1.0 + self.signed_frac))
        f = f0.copy()
        f[PSI] = applied
        if not np.all(np.isfinite(f)) or np.any(f <= 0.0):
            raise RuntimeError('Nonpositive or nonfinite signed-sweep weights')
        self.audit.append({
            'qfsweep_signed_frac': self.signed_frac,
            'qfsweep_multiplier': 1.0 + self.signed_frac,
            'qfsweep_pretrained_psi': reference,
            'qfsweep_applied_psi': applied,
            'qfsweep_ratio': applied / reference,
        })
        return (q, r, f), z
