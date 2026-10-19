"""Run: python -m unittest -v test_ac_ablation"""
import unittest

import numpy as np

import rl_nmpc as rl
from ac_ablation import (
    learn_selected_component, net_parameter_change_norm,
    net_unchanged, snapshot_net,
)


def synthetic_rollout_rows():
    rng = np.random.default_rng(20261019)
    rows = []
    for i in range(16):
        s = rng.normal(0., .12, 12)
        e = rng.normal(0., .08, 6)
        z = rng.normal(0., .3, 3)
        sn = s + rng.normal(0., .01, 12)
        en = e + rng.normal(0., .01, 6)
        rows.append((s, e, z, -float(i+1), sn, en, False))
    return rows


class ActorCriticAblationTests(unittest.TestCase):
    def test_actor_only_fixes_critic_parameters_and_adam(self):
        a = rl.Agent(seed=8, actor_lr=.001, critic_lr=.001)
        actor_before, critic_before = snapshot_net(a.actor), snapshot_net(a.critic)
        record = learn_selected_component(a, synthetic_rollout_rows(), 'actor')
        self.assertTrue(net_unchanged(a.critic, critic_before))
        self.assertFalse(net_unchanged(a.actor, actor_before))
        self.assertGreater(net_parameter_change_norm(a.actor, actor_before), 0.)
        self.assertEqual(a.actor.t, actor_before['t'] + 1)
        self.assertEqual(a.critic.t, critic_before['t'])
        self.assertEqual(record['critic_parameter_delta'], 0.)
        self.assertTrue(record['actor_update_applied'])
        self.assertFalse(record['critic_update_applied'])

    def test_critic_only_fixes_actor_parameters_and_adam(self):
        a = rl.Agent(seed=8, actor_lr=.001, critic_lr=.001)
        actor_before, critic_before = snapshot_net(a.actor), snapshot_net(a.critic)
        record = learn_selected_component(a, synthetic_rollout_rows(), 'critic')
        self.assertTrue(net_unchanged(a.actor, actor_before))
        self.assertFalse(net_unchanged(a.critic, critic_before))
        self.assertGreater(net_parameter_change_norm(a.critic, critic_before), 0.)
        self.assertEqual(a.actor.t, actor_before['t'])
        self.assertEqual(a.critic.t, critic_before['t'] + 1)
        self.assertEqual(record['actor_parameter_delta'], 0.)
        self.assertFalse(record['actor_update_applied'])
        self.assertTrue(record['critic_update_applied'])

    def test_both_mode_matches_original_learning(self):
        rows = synthetic_rollout_rows()
        expected = rl.Agent(seed=19, actor_lr=.001, critic_lr=.001)
        actual = rl.Agent(seed=19, actor_lr=.001, critic_lr=.001)
        prior = expected.learn(rows)
        current = learn_selected_component(actual, rows, 'both')
        for key, value in prior.items():
            self.assertEqual(current[key], value)
        for name in ('actor', 'critic'):
            a, b = getattr(expected, name), getattr(actual, name)
            self.assertTrue(net_unchanged(a, snapshot_net(b)))
        self.assertEqual(expected.actor.t, actual.actor.t)
        self.assertEqual(expected.critic.t, actual.critic.t)

    def test_inactive_network_remains_frozen_across_updates(self):
        for mode, inactive in [('actor', 'critic'), ('critic', 'actor')]:
            agent = rl.Agent(seed=42, actor_lr=.001, critic_lr=.001)
            snapshot = snapshot_net(getattr(agent, inactive))
            for _ in range(3):
                learn_selected_component(agent, synthetic_rollout_rows(), mode)
            self.assertTrue(net_unchanged(getattr(agent, inactive), snapshot))

    def test_invalid_mode_rejected_without_updating(self):
        a = rl.Agent(seed=2)
        old_actor = snapshot_net(a.actor)
        old_critic = snapshot_net(a.critic)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            learn_selected_component(a, synthetic_rollout_rows(), 'freeze_qfpsi')
        self.assertTrue(net_unchanged(a.actor, old_actor))
        self.assertTrue(net_unchanged(a.critic, old_critic))


if __name__ == '__main__':
    unittest.main()
