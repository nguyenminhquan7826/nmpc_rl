"""Evaluation-only *functional* isolation of the pretrained Qf_psi(s) head.

Critic Qf_psi(s) is a neural-network output with shared hidden layers.
Freezing only the last-layer column does NOT freeze this state-dependent
output when hidden layers update. This module instead retains a deep-copied
pretrained Critic and substitutes exactly its Qf_psi(s) output while all other
Critic outputs and the value baseline remain online-updated.

This experiment does NOT yield a standalone adapted checkpoint: inference
requires the original pretrained critic's frozen heading-weight function.
"""
from copy import deepcopy

import numpy as np

import rl_nmpc as rl


QFPSI_INDEX = 2


def install_pretrained_qfpsi_function(agent):
    """Replace agent.critic_terms with a masked version until caller restores it.

    All calls through Agent.propose(), Agent.value() and Agent.learn() then use
    exactly Qf_psi(s) from the original pretrained checkpoint. In the
    optimizer gradient, sig[...,2]=0 ensures dV/d(raw_output_2) and the
    Qf_psi regularizer derivative are both zero. Other six Critic outputs
    (5 Qf multipliers + value baseline) update as before.

    Return the previous bound method, plus a reference function that computes
    the frozen Qf_psi weight on arbitrary state vectors for audit.
    """
    original_critic_terms = agent.critic_terms
    pretrained_critic = deepcopy(agent.critic)

    def pretrained_qfpsi(s):
        raw, _ = pretrained_critic.forward(s)
        sig = rl.sigmoid(raw[..., QFPSI_INDEX])
        multiplier = rl.FLO + (rl.FHI - rl.FLO) * sig
        return rl.F_BASE[QFPSI_INDEX] * multiplier

    def critic_terms_with_frozen_qfpsi(s):
        mult, sig, baseline, raw = original_critic_terms(s)
        frozen_multiplier = pretrained_qfpsi(s) / rl.F_BASE[QFPSI_INDEX]

        masked_mult = np.array(mult, copy=True)
        masked_sig = np.array(sig, copy=True)
        masked_mult[..., QFPSI_INDEX] = frozen_multiplier

        # In Agent.learn(), dm/draw = (FHI-FLO)*sig*(1-sig).
        # Zero the derivative for the masked output: no gradient flows into
        # a state-dependent Qf_psi from the current network. Shared hidden
        # parameters still learn from the other Critic outputs.
        masked_sig[..., QFPSI_INDEX] = 0.0

        return masked_mult, masked_sig, baseline, raw

    agent.critic_terms = critic_terms_with_frozen_qfpsi
    return original_critic_terms, pretrained_qfpsi


def verify_frozen_qfpsi(weights, frozen_weight, atol=1e-9):
    """Audit applied controller weights, never only unexposed NN parameters."""
    qf = np.asarray(weights[2], dtype=float)
    if qf.shape != (6,) or not np.isclose(
        qf[QFPSI_INDEX], frozen_weight, rtol=1e-10, atol=atol,
    ):
        raise RuntimeError('Qf_psi(s) was not held at its pretrained value')
