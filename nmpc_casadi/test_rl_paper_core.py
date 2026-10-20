"""Regression tests for paper-defined, architecture-independent RL mathematics."""
import unittest

import numpy as np

from rl_paper_core import (
    actor_log_policy_objective,
    critic_squared_td_loss,
    paper_reward,
    project_box_admissible,
    state_and_input_increment_cost,
    td_error,
)


class PaperEquationsTests(unittest.TestCase):
    def test_reward_exact_equation_4_and_batch(self):
        e = np.array([0., 0., .2, 0., .1, -.1])
        du = np.array([.02, -.01])
        q = np.array([12., 3., 15., 6., 200., 300.])
        r = np.array([.6, 16.])
        expected = 15*.2**2 + 200*.1**2 + 300*.1**2 + .6*.02**2 + 16*.01**2
        self.assertAlmostEqual(state_and_input_increment_cost(e, du, q, r), expected)
        self.assertAlmostEqual(paper_reward(e, du, q, r), -expected)
        batch = paper_reward(np.stack([e, 2*e]), np.stack([du, 2*du]), q, r)
        np.testing.assert_allclose(batch, [-expected, -4*expected], atol=1e-12)

    def test_td_exact_equations_6_7(self):
        d = td_error(-2., -5., -4., .9)
        self.assertAlmostEqual(float(d), -.6)
        self.assertAlmostEqual(float(critic_squared_td_loss(d)), .18)
        self.assertAlmostEqual(float(td_error(-2., -5., 0., .9)), 3.)

    def test_actor_loss_is_negative_td_times_logp(self):
        td = np.array([1.25, -0.5])
        logp = np.array([-1., -2.])
        self.assertAlmostEqual(actor_log_policy_objective(logp, td), .125)
        with self.assertRaises(ValueError):
            actor_log_policy_objective(logp, np.array([1.]))

    def test_box_projection_exact_clipping(self):
        x = np.array([.1, 2.3, 99.])
        low = np.array([.2, .2, .2])
        high = np.array([1., 3., 5.])
        np.testing.assert_array_equal(project_box_admissible(x,low,high),[.2,2.3,5.])
        with self.assertRaises(ValueError):
            project_box_admissible(x, np.array([0., .2, .2]), high)

    def test_reward_rejects_non_positive_and_mismatched_weights(self):
        with self.assertRaises(ValueError):
            paper_reward(np.zeros(6), np.zeros(2), np.zeros(6), np.ones(2))
        with self.assertRaises(ValueError):
            paper_reward(np.zeros(6), np.zeros(2), np.ones(6), np.ones(3))


if __name__ == "__main__":
    unittest.main()
