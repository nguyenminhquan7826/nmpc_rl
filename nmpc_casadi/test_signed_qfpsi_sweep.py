"""Phase 2D signed Qf_psi(s) sensitivity: isolation and reproducibility.

Run: python -m unittest -v test_signed_qfpsi_sweep
"""
import unittest

import numpy as np

import rl_nmpc as rl
import adapt_nmpc as adapt
from ac_ablation import snapshot_net, net_unchanged
from signed_qfpsi_sweep import SignedTerminalPsiPolicy, OFFSETS


def make_state(seed=51):
    return np.random.default_rng(seed).normal(0., 0.2, 12)


def artificial_rows(n=16):
    rng = np.random.default_rng(1421)
    rows = []
    for j in range(n):
        s = rng.normal(0., .12, 12)
        e = rng.normal(0., .1, 6)
        z = rng.normal(0., .05, 3)
        sn = s + rng.normal(0., .02, 12)
        en = e + rng.normal(0., .02, 6)
        rows.append((s, e, z, -float(j + 1), sn, en, 0.0))
    return rows


class SignedQfPsiSweepTests(unittest.TestCase):
    def test_exact_five_sweep_points_and_positive_multiplier(self):
        a = rl.Agent(seed=74)
        s = make_state()
        assert len(OFFSETS) == 5
        for offset in OFFSETS:
            p = SignedTerminalPsiPolicy(a, offset)
            (q, r, f), z = p.propose(s, explore=False)
            original = p.terminal_weights(s)
            self.assertEqual(p.signed_frac, offset)
            self.assertGreater(f[2], 0)
            self.assertAlmostEqual(float(f[2]/original[2]), 1.0+offset, places=13)
            self.assertEqual(p.audit[-1]['qfsweep_signed_frac'], offset)
            self.assertEqual(p.audit[-1]['qfsweep_multiplier'], 1.0+offset)
            self.assertTrue(p.terminal_unchanged())

    def test_only_heading_terminal_weight_changes_at_each_state(self):
        gen = np.random.default_rng(55)
        for offset in OFFSETS:
            p = SignedTerminalPsiPolicy(rl.Agent(seed=19), offset)
            for _ in range(5):
                s = gen.normal(0., .25, 12)
                f0 = p.terminal_weights(s)
                (q, r, f), z = p.propose(s, explore=True)
                np.testing.assert_array_equal(
                    f[[0, 1, 3, 4, 5]], f0[[0, 1, 3, 4, 5]],
                )
                np.testing.assert_allclose(
                    f[2], (1.0+offset)*f0[2], rtol=1e-14, atol=1e-10,
                )
                self.assertEqual(float(f[2]),
                                 p.audit[-1]['qfsweep_applied_psi'])
                self.assertEqual(float(f0[2]),
                                 p.audit[-1]['qfsweep_pretrained_psi'])

    def test_zero_offset_full_preupdate_parity_with_split_2a(self):
        a = rl.Agent(seed=11, actor_lr=.001, critic_lr=.001)
        b = rl.Agent(seed=11, actor_lr=.001, critic_lr=.001)
        from split_critic_eval import SplitValueTerminalPolicy
        zero = SignedTerminalPsiPolicy(a, 0.0)
        original = SplitValueTerminalPolicy(b)
        for _ in range(20):
            s = make_state(20+_)
            w0, z0 = zero.propose(s, True)
            w1, z1 = original.propose(s, True)
            np.testing.assert_array_equal(z0, z1)
            for x, y in zip(w0, w1):
                np.testing.assert_array_equal(x, y)
        self.assertTrue(zero.terminal_unchanged())

    def test_signed_arms_share_exact_actor_samples_even_with_different_qf(self):
        arms = [SignedTerminalPsiPolicy(rl.Agent(seed=32), frac)
                for frac in OFFSETS]
        for k in range(16):
            s = make_state(k+80)
            actions = [a.propose(s, True) for a in arms]
            for action in actions[1:]:
                np.testing.assert_array_equal(actions[0][1], action[1])
                np.testing.assert_array_equal(actions[0][0][0], action[0][0])
                np.testing.assert_array_equal(actions[0][0][1], action[0][1])
        self.assertEqual(set(x.signed_frac for x in arms), set(OFFSETS))

    def test_value_critic_and_actor_can_learn_without_updating_terminal(self):
        agent = rl.Agent(seed=51, actor_lr=.001, critic_lr=.001)
        signed = SignedTerminalPsiPolicy(agent, -0.05)
        s = make_state(13)
        f_before = signed.terminal_weights(s).copy()
        initial_actor = snapshot_net(agent.actor)
        initial_critic = snapshot_net(agent.critic)
        for _ in range(5):
            agent.learn(artificial_rows())
        self.assertFalse(net_unchanged(agent.actor, initial_actor))
        self.assertFalse(net_unchanged(agent.critic, initial_critic))
        self.assertTrue(signed.terminal_unchanged())
        np.testing.assert_array_equal(signed.terminal_weights(s), f_before)

    def test_reject_off_sweep_nonfinite_and_out_of_bounds(self):
        for bad in (-0.10, 0.10, 0.01, -0.03, np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, 'offset'):
                    SignedTerminalPsiPolicy(rl.Agent(seed=4), bad)

    def test_optional_evaluator_roster_and_mode_guard(self):
        self.assertEqual(adapt.active_controllers(),
                         ('fixed_nmpc', 'ac_nmpc_frozen', 'ac_nmpc_online'))
        expanded = adapt.active_controllers(
            include_ablation=True, split_critic=True,
            signed_qfpsi_sweep=True,
        )
        self.assertEqual(len(expanded), 11)
        self.assertEqual(tuple(expanded[-5:]),
                         tuple(adapt.SIGNED_QFPSI_SWEEP_ARMS))
        self.assertEqual(tuple(adapt.SIGNED_QFPSI_SWEEP_ARMS.values()), OFFSETS)
        with self.assertRaisesRegex(ValueError, 'requires'):
            adapt.active_controllers(
                include_ablation=True, signed_qfpsi_sweep=True,
            )
        with self.assertRaisesRegex(ValueError, 'requires'):
            adapt.active_controllers(
                split_critic=True, signed_qfpsi_sweep=True,
            )


if __name__ == '__main__':
    unittest.main()
