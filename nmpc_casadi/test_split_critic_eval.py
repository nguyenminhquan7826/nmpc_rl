"""Phase 2A structural checks: independent Qf network, TD Value unchanged.

Run: python -m unittest -v test_split_critic_eval
"""
import unittest
import numpy as np

import rl_nmpc as rl
import adapt_nmpc as adapt
from ac_ablation import snapshot_net, net_unchanged
from split_critic_eval import (
    split_pretrained_terminal_head, SplitValueTerminalPolicy,
)


def artificial_rows(n=16, seed=141):
    gen = np.random.default_rng(seed)
    rows = []
    for j in range(n):
        s = gen.normal(0., .14, 12)
        e = gen.normal(0., .07, 6)
        z = gen.normal(0., .2, 3)
        sn = s + gen.normal(0., .015, 12)
        en = e + gen.normal(0., .01, 6)
        rows.append((s, e, z, -float(j + 1), sn, en, 0.))
    return rows


class SplitCriticArchitectureTests(unittest.TestCase):
    def test_terminal_head_exactly_reproduces_pretrained_six_outputs(self):
        a = rl.Agent(seed=49)
        gen = np.random.default_rng(4)
        state = gen.normal(0., .2, (9, 12))
        original_raw, _ = a.critic.forward(state)
        h = split_pretrained_terminal_head(a.critic)
        split_raw, _ = h.forward(state)
        self.assertEqual(split_raw.shape, (9, 6))
        np.testing.assert_allclose(split_raw, original_raw[:, :6], rtol=1e-13, atol=1e-12)
        self.assertEqual(len(h.p), len(a.critic.p))
        self.assertEqual(h.p[-2].shape[1], 6)
        self.assertEqual(h.p[-1].shape, (6,))
        for attr in ('p', 'm', 'v'):
            head = getattr(h, attr)
            critic = getattr(a.critic, attr)
            for x, y in zip(head[:-2], critic[:-2]):
                np.testing.assert_array_equal(x, y)
                self.assertFalse(np.shares_memory(x, y))
            np.testing.assert_array_equal(head[-2], critic[-2][:, :6])
            np.testing.assert_array_equal(head[-1], critic[-1][:6])
            self.assertFalse(np.shares_memory(head[-2], critic[-2]))
        self.assertEqual(h.t, a.critic.t)

    def test_pre_update_actor_samples_and_all_fourteen_weights_match(self):
        legacy = rl.Agent(seed=7, actor_lr=.001, critic_lr=.001)
        split_agent = rl.Agent(seed=7, actor_lr=.001, critic_lr=.001)
        split = SplitValueTerminalPolicy(split_agent)
        rng = np.random.default_rng(35)
        for _ in range(16):
            s = rng.normal(0., .2, 12)
            w_old, z_old = legacy.propose(s, True)
            w_new, z_new = split.propose(s, True)
            np.testing.assert_array_equal(z_old, z_new)
            np.testing.assert_array_equal(w_old[0], w_new[0])
            np.testing.assert_array_equal(w_old[1], w_new[1])
            # Six-vs-seven-output BLAS matmul can differ at roundoff.
            np.testing.assert_allclose(w_old[2], w_new[2], rtol=1e-13, atol=1e-10)
        self.assertTrue(split.terminal_unchanged())

    def test_td_critic_learns_but_terminal_network_is_isolated(self):
        a = rl.Agent(seed=74, actor_lr=.001, critic_lr=.001)
        split = SplitValueTerminalPolicy(a)
        s = np.random.default_rng(22).normal(0., .15, 12)
        qf_before = split.terminal_weights(s).copy()
        value_before = snapshot_net(a.critic)
        for _ in range(7):
            metrics = a.learn(artificial_rows())
            self.assertTrue(all(
                isinstance(v, (int, float, np.integer, np.floating))
                for v in metrics.values()
            ))
        self.assertFalse(net_unchanged(a.critic, value_before))
        self.assertTrue(split.terminal_unchanged())
        np.testing.assert_array_equal(qf_before, split.terminal_weights(s))
        self.assertGreater(np.linalg.norm(
            a.critic.forward(s)[0][:6] -
            split.terminal_head.forward(s)[0]
        ), 1e-8)

    def test_score_function_actor_still_learns_independently(self):
        a = rl.Agent(seed=41, actor_lr=.001, critic_lr=.001)
        split = SplitValueTerminalPolicy(a)
        prior = snapshot_net(a.actor)
        for _ in range(5):
            a.learn(artificial_rows())
        self.assertFalse(net_unchanged(a.actor, prior))
        self.assertTrue(split.terminal_unchanged())

    def test_validator_rejects_wrong_shape(self):
        a = rl.Agent(seed=2)
        a.critic.p[-1] = a.critic.p[-1][:6]
        with self.assertRaisesRegex(ValueError, '7 outputs'):
            split_pretrained_terminal_head(a.critic)

    def test_evaluator_roster_and_phase_guard(self):
        baseline = adapt.active_controllers()
        self.assertEqual(baseline, (
            'fixed_nmpc', 'ac_nmpc_frozen', 'ac_nmpc_online'
        ))
        expanded = adapt.active_controllers(
            include_ablation=True, split_critic=True,
        )
        self.assertIn(adapt.SPLIT_CRITIC_FROZEN_TERMINAL, expanded)
        self.assertEqual(len(expanded), 6)
        self.assertEqual(
            expanded[-1], 'ac_nmpc_split_value_frozen_qf'
        )
        with self.assertRaisesRegex(ValueError, 'requires'):
            adapt.active_controllers(split_critic=True)


if __name__ == '__main__':
    unittest.main()
