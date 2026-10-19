"""Phase 2B Qfpsi score-function experimental control unit tests.

Run: python -m unittest -v test_split_terminal_pg
"""
import unittest
import numpy as np

import rl_nmpc as rl
import adapt_nmpc as adapt
from ac_ablation import snapshot_net, net_unchanged
from split_terminal_pg import TerminalPsiScorePolicy


def agent_and_policy(seed=54, learn=True, qf_seed=20261019, band=.05):
    agent = rl.Agent(seed=seed, actor_lr=.001, critic_lr=.001)
    policy = TerminalPsiScorePolicy(
        agent, qf_seed=qf_seed, sigma=.02, lr=.0001,
        trust_frac=band, trust_reg=.1, terminal_learning=learn,
    )
    return agent, policy


def collect_batch(policy, n=16, seed=152):
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(n):
        s = rng.normal(0., .12, 12)
        e = rng.normal(0., .06, 6)
        (q, r, qf), actor_z = policy.propose(s, explore=True)
        sn = s + rng.normal(0., .01, 12)
        en = e + rng.normal(0., .008, 6)
        rows.append((
            s.copy(), e.copy(), actor_z.copy(), -1. - j*.11,
            sn, en, 0. if j+1 < n else 1.,
        ))
    return rows


class QfScoreFunctionTests(unittest.TestCase):
    def test_independent_qf_rng_does_not_shift_actor_samples(self):
        agent, policy = agent_and_policy()
        other = rl.Agent(seed=54, actor_lr=.001, critic_lr=.001)
        state = np.random.default_rng(29).normal(0., .2, 12)
        for _ in range(16):
            w, z = policy.propose(state, True)
            w_old, z_old = other.propose(state, True)
            np.testing.assert_array_equal(z, z_old)
            np.testing.assert_array_equal(w[0], w_old[0])
            np.testing.assert_array_equal(w[1], w_old[1])
        self.assertTrue(policy.terminal_unchanged())

    def test_stochastic_paired_arms_match_exactly_preupdate(self):
        _, frozen = agent_and_policy(seed=81, learn=False)
        _, learned = agent_and_policy(seed=81, learn=True)
        rg = np.random.default_rng(32)
        for _ in range(16):
            s = rg.normal(0., .1, 12)
            a, za = frozen.propose(s, True)
            b, zb = learned.propose(s, True)
            np.testing.assert_array_equal(za, zb)
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)
        self.assertTrue(frozen.terminal_unchanged())
        self.assertTrue(learned.terminal_unchanged())

    def test_trust_applies_to_solver_only_and_preserves_five_components(self):
        _, policy = agent_and_policy(band=.05)
        rg = np.random.default_rng(37)
        for _ in range(25):
            s = rg.normal(0., .3, 12)
            (q, r, f), z = policy.propose(s, True)
            base = policy.initial_terminal.forward(s)[0]
            ref = rl.F_BASE * (rl.FLO + (rl.FHI - rl.FLO) * rl.sigmoid(base))
            np.testing.assert_array_equal(
                f[[0, 1, 3, 4, 5]], ref[[0, 1, 3, 4, 5]],
            )
            self.assertGreaterEqual(f[2] / ref[2], .95-1e-12)
            self.assertLessEqual(f[2] / ref[2], 1.05+1e-12)
            self.assertAlmostEqual(f[2], policy.sample_audit[-1]['qfpg_applied_psi'])
            self.assertTrue(np.all(np.isfinite(f)))

    def test_terminal_policy_gradient_updates_only_psi_output(self):
        agent, policy = agent_and_policy()
        value_before = snapshot_net(agent.critic)
        actor_before = snapshot_net(agent.actor)
        rows = collect_batch(policy)
        m = policy.update_terminal(rows)
        self.assertEqual(m['qfpg_training_enabled'], 1.)
        self.assertEqual(policy.qf_training_updates, 1)
        self.assertGreater(m['qfpg_parameter_delta'], 0.)
        self.assertFalse(policy.terminal_unchanged())
        self.assertTrue(policy.inactive_weights_unchanged())
        self.assertTrue(policy.initial_function_unchanged_for_other_five(
            np.asarray([r[0] for r in rows]),
        ))
        self.assertTrue(net_unchanged(agent.actor, actor_before))
        self.assertTrue(net_unchanged(agent.critic, value_before))
        # Value TD update afterwards does not touch terminal net.
        terminal_snapshot = [a.copy() for a in policy.terminal_head.p]
        agent.learn(rows)
        for old, new in zip(terminal_snapshot, policy.terminal_head.p):
            np.testing.assert_array_equal(old, new)

    def test_frozen_stochastic_control_never_updates_terminal_net(self):
        agent, policy = agent_and_policy(learn=False)
        rows = collect_batch(policy)
        m = policy.update_terminal(rows)
        self.assertEqual(m['qfpg_training_enabled'], 0.)
        self.assertEqual(m['qfpg_updates'], 0.)
        self.assertTrue(policy.terminal_unchanged())
        self.assertTrue(policy.inactive_weights_unchanged())
        agent.learn(rows)
        self.assertTrue(policy.terminal_unchanged())

    def test_terminal_optimizer_resets_old_critic_adam_momentum(self):
        agent = rl.Agent(seed=71, actor_lr=.001, critic_lr=.001)
        # Simulate a trained Critic, with Adam moments and nonzero step.
        synth = collect_batch(agent_and_policy()[1])
        agent.learn(synth)
        self.assertGreater(agent.critic.t, 0)
        policy = TerminalPsiScorePolicy(
            agent, qf_seed=2, sigma=.02, lr=.0001,
            trust_frac=.05, trust_reg=.1, terminal_learning=True,
        )
        self.assertEqual(policy.terminal_head.t, 0)
        self.assertTrue(all(np.all(x == 0) for x in policy.terminal_head.m))
        self.assertTrue(all(np.all(x == 0) for x in policy.terminal_head.v))
        states = np.random.default_rng(33).normal(0., .08, (2, 12))
        np.testing.assert_allclose(
            policy.terminal_head.forward(states)[0],
            agent.critic.forward(states)[0][:, :6],
            rtol=1e-13, atol=1e-12,
        )

    def test_requires_sampling_alignment(self):
        _, p = agent_and_policy()
        with self.assertRaisesRegex(RuntimeError, 'misaligned'):
            p.update_terminal([])
        rows = collect_batch(p, n=5)
        with self.assertRaisesRegex(RuntimeError, 'misaligned'):
            p.update_terminal(rows[:-1])
        self.assertEqual(len(p.pending), 5)

    def test_parameter_validation(self):
        for key, bad in (
            ('sigma', 0.),
            ('sigma', float('nan')),
            ('lr', -1.),
            ('trust_frac', .3),
            ('trust_reg', -1.),
            ('qf_seed', -1),
        ):
            args = dict(qf_seed=10, sigma=.02, lr=.0001,
                        trust_frac=.05, trust_reg=.1,
                        terminal_learning=True)
            args[key] = bad
            with self.subTest(key=key, bad=bad):
                with self.assertRaises(ValueError):
                    TerminalPsiScorePolicy(rl.Agent(seed=1), **args)

    def test_roster_is_explicit_and_legacy_unmodified(self):
        self.assertEqual(adapt.active_controllers(), (
            'fixed_nmpc', 'ac_nmpc_frozen', 'ac_nmpc_online'
        ))
        self.assertEqual(len(adapt.active_controllers(
            include_ablation=True, split_critic=True, split_qf_pg=True,
        )), 8)
        with self.assertRaises(ValueError):
            adapt.active_controllers(
                include_ablation=True, split_qf_pg=True,
            )


if __name__ == '__main__':
    unittest.main()
