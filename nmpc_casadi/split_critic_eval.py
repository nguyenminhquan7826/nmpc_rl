"""Evaluation-only Phase 2A: independent value and terminal-weight networks.

The saved R3A Critic has 7 outputs and a state/value representation
V(s,e) = -softplus(raw_6) - e^T Qf_value(s) e.
It is still TD-Huber trained as the VALUE network via the existing
Agent.learn(), including legacy Qf-related internal basis parameters.

A genuinely independent, 6-output terminal-weight network is split by
copying the checkpoint's first SIX output columns and all hidden layers.
It drives NMPC Qf(s) but is NOT trained in Phase 2A, ensuring that no
TD gradient can reach NMPC terminal weights.

This stage intentionally does not pretend that a Qf-control loss exists.
It is a structural-isolation acceptance test before introducing a
separate Qf control-performance learning objective in Phase 2B.

NOT a formal MPC safety certificate. A normal Agent checkpoint does not
replay the composite policy. Keep pretrained assets unchanged.
"""
from copy import deepcopy

import numpy as np
import rl_nmpc as rl


def split_pretrained_terminal_head(pretrained_critic):
    """Return a fully independent six-output MLP identical to Qf(s) at load.

    Copying the hidden layers and slicing each last-layer parameter/moment
    ensures the exact INITIAL pretrained function, including Adam state.
    Value Critic remains a distinct seven-output network, including its
    own independent six internal quadratic coefficients and value baseline.
    """
    head = deepcopy(pretrained_critic)
    if head.p[-2].shape[-1] != 7 or head.p[-1].shape != (7,):
        raise ValueError('Pretrained Critic must have exactly 7 outputs')
    for attr in ('p', 'm', 'v'):
        parts = getattr(head, attr)
        parts[-2] = parts[-2][:, :6].copy()
        parts[-1] = parts[-1][:6].copy()
    return head


class SplitValueTerminalPolicy:
    """Policy Q/R comes from existing Actor; Qf comes from a separate head.

    The original Agent.critic is retained as the VALUE critic, and is
    updated by unchanged Agent.learn() on every rollout batch.
    The independent terminal head is NOT trained in Phase 2A.
    """
    def __init__(self, agent):
        self.terminal_head = split_pretrained_terminal_head(agent.critic)
        self.initial_terminal = deepcopy(self.terminal_head)
        self.agent = agent

    def terminal_weights(self, s):
        raw, _ = self.terminal_head.forward(s)
        return rl.F_BASE * (
            rl.FLO + (rl.FHI - rl.FLO) * rl.sigmoid(raw)
        )

    def propose(self, s, explore):
        # Mirrors rl.Agent.propose's Actor math exactly, with same sampled z.
        mu, _ = self.agent.actor.forward(s)
        z = mu + (
            self.agent.rng.normal(0., self.agent.sigma, 3)
            if explore else 0
        )
        a = rl.LO + (rl.HI - rl.LO) * (np.tanh(z) + 1.) / 2.
        q = rl.Q_BASE.copy()
        q[2] *= a[1]
        q[4:] *= a[0]
        r = rl.R_BASE.copy()
        r[1] *= a[2]
        return (q, r, self.terminal_weights(s)), z

    def terminal_unchanged(self):
        for attr in ('p', 'm', 'v'):
            aa = getattr(self.terminal_head, attr)
            bb = getattr(self.initial_terminal, attr)
            if len(aa) != len(bb):
                return False
            if not all(np.array_equal(x, y) for x, y in zip(aa, bb)):
                return False
        return self.terminal_head.t == self.initial_terminal.t
