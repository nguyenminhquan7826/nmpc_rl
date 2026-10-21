"""Run: python -m unittest -v test_ac_ablation"""
import unittest

import numpy as np

import rl_nmpc as rl
import adapt_nmpc as adapt
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
    def test_selection_adds_arms_without_changing_default(self):
        self.assertEqual(
            adapt.active_controllers(),
            ('fixed_nmpc', 'ac_nmpc_frozen', 'ac_nmpc_online'),
        )
        self.assertEqual(
            adapt.active_controllers(include_ablation=True),
            ('fixed_nmpc', 'ac_nmpc_frozen',
             'ac_nmpc_actor_only', 'ac_nmpc_critic_only',
             'ac_nmpc_online'),
        )
        self.assertIn(
            'ac_nmpc_frozen_deterministic',
            adapt.active_controllers(
                include_frozen_deterministic=True, include_ablation=True,
            ),
        )

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
        self.assertTrue(all(np.isscalar(x) and isinstance(
            x, (int, float, np.integer, np.floating, np.bool_)
        ) for x in record.values()))
        self.assertNotIn('train_component', record)
        self.assertEqual(
            set(record), set(rl.Agent(seed=8).learn(synthetic_rollout_rows())),
        )

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
        self.assertTrue(all(np.isscalar(x) and isinstance(
            x, (int, float, np.integer, np.floating, np.bool_)
        ) for x in record.values()))
        self.assertNotIn('train_component', record)

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

    def test_rollout_weighted_aggregation_accepts_every_ablation_metric(self):
        # Regression for rl_nmpc.rollout line 692: every Agent.learn
        # diagnostic is averaged, not only the documented metric subset.
        rows = synthetic_rollout_rows()
        for mode in ('actor', 'critic', 'both'):
            a = rl.Agent(seed=18, actor_lr=.001, critic_lr=.001)
            first = learn_selected_component(a, rows, mode)
            second = learn_selected_component(a, rows, mode)
            self.assertEqual(set(first), set(second))
            for metric in first:
                averaged = float(np.average(
                    [first[metric], second[metric]], weights=[16., 7.]
                ))
                self.assertTrue(np.isfinite(averaged),
                                f'{mode}: {metric} is not numeric')

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
