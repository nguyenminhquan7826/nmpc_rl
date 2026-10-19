"""Test NMPC-input-only Qf_psi trust projection and unchanged Critic learning.

Run: python -m unittest -v test_qfpsi_trust_region
"""
import unittest
import numpy as np

import adapt_nmpc as adapt
import rl_nmpc as rl
from ac_ablation import snapshot_net, net_unchanged
from qfpsi_trust_region import QfPsiTrustProjector


def synthetic_rows(n=16, seed=1024):
    g = np.random.default_rng(seed)
    rows = []
    for j in range(n):
        state = g.normal(0., .15, 12)
        error = g.normal(0., .07, 6)
        action_z = g.normal(0., .2, 3)
        next_state = state + g.normal(0., .015, 12)
        next_error = error + g.normal(0., .01, 6)
        rows.append((state, error, action_z, -float(j+1),
                     next_state, next_error, 0.))
    return rows


class QfPsiTrustProjectionTests(unittest.TestCase):
    def setUp(self):
        self.agent = rl.Agent(seed=17, actor_lr=.001, critic_lr=.001)
        self.state = np.random.default_rng(18).normal(0., .14, 12)

    def test_zero_pct_equals_pretrained_function_at_same_state(self):
        projector = QfPsiTrustProjector(self.agent.critic, 0.0)
        (q, r, f), sampled_z = self.agent.propose(self.state, explore=True)
        before_q, before_r, before_f = q.copy(), r.copy(), f.copy()
        weights, log = projector.project((q, r, f), self.state)
        self.assertAlmostEqual(weights[2][2], float(projector.reference_weight(self.state)))
        self.assertAlmostEqual(log['qfpsi_applied_ratio'], 1.0)
        np.testing.assert_array_equal(q, before_q)
        np.testing.assert_array_equal(r, before_r)
        np.testing.assert_array_equal(f, before_f)
        np.testing.assert_array_equal(weights[0], q)
        np.testing.assert_array_equal(weights[1], r)
        np.testing.assert_array_equal(weights[2][[0,1,3,4,5]], f[[0,1,3,4,5]])

    def test_band_limits_ratio_and_preserves_other_weights(self):
        projector = QfPsiTrustProjector(self.agent.critic, .10)
        q, r, f = self.agent.propose(self.state, False)[0]
        ref = float(projector.reference_weight(self.state))
        high = f.copy()
        high[2] = ref * 4.
        weights, audit = projector.project((q, r, high), self.state)
        self.assertAlmostEqual(weights[2][2], ref*1.10)
        self.assertEqual(audit['qfpsi_projected'], 1)
        self.assertAlmostEqual(audit['qfpsi_raw_ratio'], 4.)
        np.testing.assert_array_equal(weights[2][[0,1,3,4,5]], high[[0,1,3,4,5]])
        low = f.copy()
        low[2] = ref*.25
        weights, audit = projector.project((q, r, low), self.state)
        self.assertAlmostEqual(weights[2][2], ref*.90)
        self.assertEqual(audit['qfpsi_projected'], 1)

    def test_checkpoint_reference_does_not_follow_updated_critic(self):
        projector = QfPsiTrustProjector(self.agent.critic, .05)
        frozen_qf = projector.reference_weight(self.state)
        before = snapshot_net(self.agent.critic)
        for _ in range(4):
            self.agent.learn(synthetic_rows())
        self.assertFalse(net_unchanged(self.agent.critic, before))
        self.assertEqual(float(projector.reference_weight(self.state)), float(frozen_qf))
        q, r, f = self.agent.propose(self.state, False)[0]
        (qq, rr, projected), audit = projector.project((q, r, f), self.state)
        self.assertGreaterEqual(audit['qfpsi_applied_ratio'], .95-1e-10)
        self.assertLessEqual(audit['qfpsi_applied_ratio'], 1.05+1e-10)

    def test_projector_does_not_change_td_learning_or_adam(self):
        # Fixed transitions + common RNG: projection only affects NMPC action,
        # not the Agent.learn() gradient or state. The plant will differ once
        # deployed; this test deliberately isolates the method call.
        with_projection = rl.Agent(seed=28, actor_lr=.001, critic_lr=.001)
        without_projection = rl.Agent(seed=28, actor_lr=.001, critic_lr=.001)
        proj = QfPsiTrustProjector(with_projection.critic, .10)
        rows = synthetic_rows()
        for j in range(3):
            weights, z = with_projection.propose(self.state, True)
            base_weights, base_z = without_projection.propose(self.state, True)
            np.testing.assert_array_equal(z, base_z)
            for x, y in zip(weights, base_weights):
                np.testing.assert_array_equal(x, y)
            proj.project(weights, self.state)
            metrics_a = with_projection.learn(rows)
            metrics_b = without_projection.learn(rows)
            self.assertEqual(metrics_a, metrics_b)
        self.assertTrue(adapt.network_state_unchanged(
            with_projection, adapt.snapshot_network_state(without_projection)
        ))

    def test_projection_rejects_invalid_parameters(self):
        for bad in [float('nan'), -0.1, 1.1, float('inf')]:
            with self.assertRaises(ValueError):
                QfPsiTrustProjector(self.agent.critic, bad)
        proj = QfPsiTrustProjector(self.agent.critic, .1)
        q, r, f = self.agent.propose(self.state, False)[0]
        bad = f.copy()
        bad[2] = float('nan')
        with self.assertRaises(ValueError):
            proj.project((q, r, bad), self.state)
        with self.assertRaises(ValueError):
            proj.project((q, r, f[:5]), self.state)

    def test_explicit_modes_preserve_default_comparators(self):
        default = adapt.active_controllers()
        self.assertEqual(default, (
            'fixed_nmpc', 'ac_nmpc_frozen', 'ac_nmpc_online',
        ))
        roster = adapt.active_controllers(
            include_ablation=True, qfpsi_trust_frac=.10,
        )
        self.assertEqual(len(roster), 7)
        self.assertIn('ac_nmpc_critic_qfpsi_trust', roster)
        self.assertIn('ac_nmpc_online_qfpsi_trust', roster)
        with self.assertRaises(ValueError):
            adapt.active_controllers(qfpsi_trust_frac=.1)


if __name__ == '__main__':
    unittest.main()
