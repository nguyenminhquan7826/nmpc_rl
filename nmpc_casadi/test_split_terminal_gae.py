"""Phase 2C GAE unit tests for isolated terminal Qf_psi policy gradient.

Run: python -m unittest -v test_split_terminal_gae
"""
import unittest
import numpy as np

import rl_nmpc as rl
import adapt_nmpc as adapt
from ac_ablation import snapshot_net, net_unchanged
from split_terminal_pg import TerminalPsiScorePolicy, generalized_advantage


def policy_and_agent(seed=73, gae_lambda=0.95):
    agent = rl.Agent(seed=seed, actor_lr=.001, critic_lr=.001)
    policy = TerminalPsiScorePolicy(
        agent, qf_seed=20261019, sigma=.02, lr=.0001,
        trust_frac=.05, trust_reg=.1, terminal_learning=True,
        gae_lambda=gae_lambda,
    )
    return agent, policy


def sample_batch(policy, n=16):
    rng = np.random.default_rng(1234)
    rows = []
    for j in range(n):
        s = rng.normal(0., .12, 12)
        e = rng.normal(0., .07, 6)
        weights, z = policy.propose(s, True)
        sn = s + rng.normal(0., .01, 12)
        en = e + rng.normal(0., .015, 6)
        rows.append((s.copy(), e.copy(), z.copy(),
                     float(-1.0 - .1*j), sn, en,
                     float(j == n-1)))
    return rows


class TerminalGaeTests(unittest.TestCase):
    def test_zero_lambda_exactly_matches_old_one_step_td(self):
        rng = np.random.default_rng(15)
        for n in (1, 2, 16, 31):
            r = rng.normal(size=n)
            v = rng.normal(size=n)
            vn = rng.normal(size=n)
            terminated = rng.choice([0., 1.], size=n)
            adv, delta = generalized_advantage(
                r, v, vn, terminated, gamma=rl.GAMMA, gae_lambda=0.0,
            )
            expected = r + rl.GAMMA*(1. - terminated)*vn - v
            np.testing.assert_array_equal(delta, expected)
            np.testing.assert_array_equal(adv, expected)
            normalized_adv = adv / max(float(np.std(adv)), 1.)
            normalized_old = expected / max(float(np.std(expected)), 1.)
            np.testing.assert_array_equal(normalized_adv, normalized_old)

    def test_hand_computed_multistep_gae(self):
        rewards = np.array([-2., -3., -5.])
        v = np.array([-2.5, -4., -1.])
        vn = np.array([-4., -1., 99.])
        terminated = np.array([0., 0., 1.])
        gae, delta = generalized_advantage(
            rewards, v, vn, terminated, gamma=.9, gae_lambda=.7,
        )
        expected_delta = np.array([-3.1, .1, -4.])
        np.testing.assert_allclose(delta, expected_delta, rtol=0, atol=1e-12)
        expected_gae = np.array([
            -3.1 + .63 * (.1 + .63 * -4.),
            .1 + .63 * -4., -4.,
        ])
        np.testing.assert_allclose(gae, expected_gae, rtol=0, atol=1e-12)

    def test_true_terminal_blocks_advantage_propagation(self):
        r = np.array([1., 2., 3., 4.])
        v = np.zeros(4)
        vn = np.zeros(4)
        term = np.array([0., 1., 0., 0.])
        adv, td = generalized_advantage(
            r, v, vn, term, gamma=.9, gae_lambda=1.,
        )
        np.testing.assert_allclose(adv, [2.8, 2., 6.6, 4.])

    def test_batch_truncation_bootstraps_without_fake_terminal(self):
        r = np.array([-1., -2.])
        v = np.array([0., 0.])
        vn = np.array([0., 10.])
        nonterminal = np.array([0., 0.])
        adv, delta = generalized_advantage(
            r, v, vn, nonterminal, gamma=.9, gae_lambda=.8,
        )
        # Last delta MUST include next_V at a nonterminal batch boundary.
        np.testing.assert_allclose(delta, [-1., 7.])
        np.testing.assert_allclose(adv, [-1. + .72*7., 7.])
        adv_terminal, delta_terminal = generalized_advantage(
            r, v, vn, [0., 1.], gamma=.9, gae_lambda=.8,
        )
        np.testing.assert_allclose(delta_terminal, [-1., -2.])
        np.testing.assert_allclose(adv_terminal, [-1. + .72*-2., -2.])

    def test_same_actor_qf_samples_before_update_as_one_step(self):
        a, pg = policy_and_agent(seed=44, gae_lambda=0.)
        b, gae = policy_and_agent(seed=44, gae_lambda=.95)
        rng = np.random.default_rng(62)
        for _ in range(16):
            s = rng.normal(0., .15, 12)
            w0, z0 = pg.propose(s, True)
            w1, z1 = gae.propose(s, True)
            np.testing.assert_array_equal(z0, z1)
            for x, y in zip(w0, w1):
                np.testing.assert_array_equal(x, y)
        self.assertTrue(pg.terminal_unchanged())
        self.assertTrue(gae.terminal_unchanged())

    def test_gae_updates_only_heading_output_not_value_or_actor(self):
        agent, policy = policy_and_agent()
        critic_before = snapshot_net(agent.critic)
        actor_before = snapshot_net(agent.actor)
        rows = sample_batch(policy)
        m = policy.update_terminal(rows)
        self.assertEqual(m['qfpg_gae_lambda'], .95)
        self.assertNotEqual(m['qfpg_advantage_rms'], m['qfpg_control_td_rms'])
        self.assertGreater(m['qfpg_parameter_delta'], 0)
        self.assertFalse(policy.terminal_unchanged())
        self.assertTrue(policy.inactive_weights_unchanged())
        self.assertTrue(policy.initial_function_unchanged_for_other_five(
            np.asarray([r[0] for r in rows])
        ))
        self.assertTrue(net_unchanged(agent.critic, critic_before))
        self.assertTrue(net_unchanged(agent.actor, actor_before))
        terminal_params = [x.copy() for x in policy.terminal_head.p]
        agent.learn(rows)
        for old, new in zip(terminal_params, policy.terminal_head.p):
            np.testing.assert_array_equal(old, new)

    def test_invalid_lambda_fails_and_evaluator_is_opt_in(self):
        for bad in (-.001, 1.001, np.nan, np.inf):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, 'GAE lambda'):
                    policy_and_agent(gae_lambda=bad)
                with self.assertRaisesRegex(ValueError, 'GAE lambda'):
                    generalized_advantage([-1.], [-1.], [-1.], [0.],
                                          gae_lambda=bad)
        self.assertEqual(
            adapt.active_controllers(),
            ('fixed_nmpc', 'ac_nmpc_frozen', 'ac_nmpc_online'),
        )
        roster = adapt.active_controllers(
            include_ablation=True, split_critic=True,
            split_qf_pg=True, split_qf_gae=True,
        )
        self.assertEqual(len(roster), 9)
        self.assertEqual(roster[-1], adapt.SPLIT_QF_GAE)
        with self.assertRaisesRegex(ValueError, 'requires'):
            adapt.active_controllers(include_ablation=True,
                                     split_critic=True, split_qf_gae=True)


if __name__ == '__main__':
    unittest.main()
