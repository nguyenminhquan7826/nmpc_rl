"""R2-A regression tests: only Critic n-step TD objective changes to Huber.

Run in nmpc_casadi:
    python -m unittest -v test_reward_r1 test_critic_huber_r2a
"""
import unittest

import numpy as np

from rl_nmpc import (
    Agent, CRITIC_HUBER_DELTA, CRITIC_TD_LOSS_KIND, REWARD_SCHEME,
    huber_td_loss_and_grad,
)


class CriticHuberR2ATests(unittest.TestCase):
    def test_config_and_reward_unchanged(self):
        self.assertEqual(CRITIC_TD_LOSS_KIND, "huber")
        self.assertEqual(CRITIC_HUBER_DELTA, 50.0)
        self.assertEqual(REWARD_SCHEME, "fixed_eval_progress_failure_v1")
        meta = Agent(seed=0)._meta()
        self.assertEqual(meta["version"], 11)
        self.assertEqual(meta["critic_td_loss_kind"], "huber")
        self.assertEqual(meta["critic_huber_delta"], 50.0)

    def test_quadratic_branch(self):
        td = np.array([-10.0, 0.0, 10.0, -50.0, 50.0])
        actual, gradient = huber_td_loss_and_grad(td)
        np.testing.assert_allclose(actual, 0.5 * td * td)
        np.testing.assert_allclose(gradient, td)

    def test_linear_branch_and_very_large_td(self):
        td = np.array([-3000.0, -80.0, 80.0, 3000.0])
        actual, gradient = huber_td_loss_and_grad(td, delta=50.0)
        np.testing.assert_allclose(actual, 50.0*(np.abs(td) - 25.0))
        np.testing.assert_allclose(gradient, [-50.0, -50.0, 50.0, 50.0])
        self.assertLess(actual[0], 0.5 * td[0] ** 2)

    def test_continuity_at_threshold(self):
        eps = 1e-7
        left, _ = huber_td_loss_and_grad(np.array([50.0-eps]))
        right, _ = huber_td_loss_and_grad(np.array([50.0+eps]))
        self.assertLess(abs(float(left[0]-right[0])), 2e-5)

    def test_numeric_derivative_sign_and_scale(self):
        target = -3000.0
        for value in [-3050.0, -3002.0, -2800.0]:
            step = 1e-4

            def objective(v):
                return float(huber_td_loss_and_grad(
                    np.array([target-v]), delta=50.0
                )[0][0])

            numerical = (objective(value+step)-objective(value-step))/(2*step)
            td_slope = huber_td_loss_and_grad(np.array([target-value]))[1][0]
            self.assertAlmostEqual(numerical, -td_slope, places=5)

    def test_bad_huber_delta_rejected(self):
        for bad in [0.0, -1.0, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                huber_td_loss_and_grad(np.array([1.]), delta=bad)

    def test_critic_training_logs_huber_and_original_mse(self):
        agent = Agent(seed=7, critic_nsteps=16)
        rows = []
        for i, reward in enumerate([-1.0, -2.0, -3000.0]):
            s = np.zeros(12)
            s[0] = 0.1*i
            e = np.array([0.01, 0.0, 0.01, 0.0, 0.01, 0.01])
            _, z = agent.propose(s, explore=True)
            sn = s.copy()
            en = e.copy()
            rows.append((s, e, z, reward, sn, en, float(i == 2)))

        metrics = agent.learn(rows)
        self.assertGreater(metrics["critic_td_mse_loss"], metrics["critic_td_huber_loss"])
        self.assertEqual(metrics["critic_td_loss"], metrics["critic_td_mse_loss"])
        self.assertEqual(metrics["critic_loss"], metrics["critic_td_huber_loss"])
        self.assertAlmostEqual(
            metrics["critic_loss_total"],
            metrics["critic_td_huber_loss"] + metrics["critic_qf_reg_loss"],
        )
        self.assertGreater(metrics["critic_huber_linear_pct"], 0.0)
        self.assertGreater(metrics["critic_td_abs_max"], 2000.0)
        self.assertTrue(all(np.all(np.isfinite(p)) for p in agent.actor.p))
        self.assertTrue(all(np.all(np.isfinite(p)) for p in agent.critic.p))


if __name__ == "__main__":
    unittest.main()
