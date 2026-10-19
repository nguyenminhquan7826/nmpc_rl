"""Evaluation-only Actor/Critic update ablations.

Training code (rl_nmpc.py) is not changed. The inactive network's .update()
call is replaced *temporarily* by a no-op during Agent.learn(rows). This
preserves the exact frozen weights, Adam moments and step counter; the active
network follows its original learning rule and frozen bootstrap convention.

Important: Agent.learn() still computes both gradients. Reported learning
wall time in these diagnostic ablations includes this unused computation;
it must NOT be interpreted as production optimized Actor-only/Critic-only WCET.
"""
from __future__ import annotations

import numpy as np


MODES = ('both', 'actor', 'critic')


def snapshot_net(net):
    """Copy all trainable and optimizer state (not RNG)."""
    return {
        'p': [a.copy() for a in net.p],
        'm': [a.copy() for a in net.m],
        'v': [a.copy() for a in net.v],
        't': int(net.t),
    }


def net_unchanged(net, snapshot):
    return (
        int(net.t) == snapshot['t']
        and all(
            len(getattr(net, attr)) == len(snapshot[attr])
            and all(np.array_equal(a, b) for a, b
                    in zip(getattr(net, attr), snapshot[attr]))
            for attr in ('p', 'm', 'v')
        )
    )


def net_parameter_change_norm(net, snapshot):
    return float(np.sqrt(sum(
        np.sum((a - b) ** 2)
        for a, b in zip(net.p, snapshot['p'])
    )))


def _noop_adam_update(grads, lr):
    """Record zero applied update; do not touch Adam moments/counter."""
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError('Invalid learning rate for ablation')
    norm = float(np.sqrt(sum(
        np.sum(np.asarray(g, dtype=float) ** 2) for g in grads
    )))
    return {
        'grad_norm': norm,
        'clip_factor': 0.0,       # Update DISABLED, not clipped.
        'parameter_delta': 0.0,
    }


def learn_selected_component(agent, rows, mode='both', learn_fn=None):
    """Use the original Agent.learn() with one network's Adam step disabled.

    'both' calls Agent.learn with no wrapping and must reproduce legacy Online
    bit-for-bit for the same agent and rollouts. Actor-only uses the fixed
    pretrained Critic's value function throughout; Critic-only keeps the Actor
    policy parameters fixed but continues sampling from its Gaussian policy.
    """
    if mode not in MODES:
        raise ValueError(f'Unknown Actor/Critic ablation mode: {mode!r}')
    original_learn = agent.learn if learn_fn is None else learn_fn
    if mode == 'both':
        return original_learn(rows)

    inactive = 'critic' if mode == 'actor' else 'actor'
    net = getattr(agent, inactive)
    frozen = snapshot_net(net)
    original_update = net.update
    try:
        net.update = _noop_adam_update
        result = original_learn(rows)
    finally:
        net.update = original_update

    if not net_unchanged(net, frozen):
        raise RuntimeError(
            f'Ablation violated inactive {inactive} weights or Adam state'
        )
    # rl_nmpc.rollout averages EVERY Agent.learn() return field with
    # np.average(...). Keep the exact original numeric diagnostics schema;
    # storing a string (e.g. 'train_component') here crashes that aggregator.
    # The mode and inactive-network assertions are logged by adapt_nmpc.py.
    if any(
        not np.isscalar(value) or
        not isinstance(value, (int, float, np.integer, np.floating, np.bool_))
        for value in result.values()
    ):
        raise TypeError('Agent.learn must return numeric scalar diagnostics')
    return result
