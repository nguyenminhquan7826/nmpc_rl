"""Unit checks for reward scheme R1 (no simulation run required).

Run from nmpc_casadi:
    python -m unittest -v test_reward_r1
"""
import unittest
import numpy as np

from rl_nmpc import (
    GAMMA, Q_EVAL, R_EVAL, REWARD_SCHEME, classify_termination,
    failure_cost_at_progress, n_step_targets, reward_breakdown,
)


class RewardR1Tests(unittest.TestCase):
    def test_progress_penalty(self):
        for progress, expected in [(0., 3000.), (.5, 2000.), (1., 1000.),
                                   (-1., 3000.), (3., 1000.)]:
            self.assertAlmostEqual(failure_cost_at_progress(progress), expected)
        with self.assertRaises(ValueError):
            failure_cost_at_progress(float("nan"))

    def test_nominal_reward_unchanged(self):
        e = np.array([.1, 0, .02, 0, .03, 0])
        du = np.array([.1, .01])
        c = reward_breakdown(e, du, True, False, .5)
        want = -(np.sum(Q_EVAL * e*e) + np.sum(R_EVAL * du*du))
        self.assertAlmostEqual(c["reward"], want)
        self.assertEqual(c["terminal_failure_cost"], 0.)

    def test_solver_failure_without_termination(self):
        c = reward_breakdown(np.zeros(6), np.zeros(2), False, False, .25)
        self.assertEqual(c["reward"], -10.)

    def test_early_terminal_failure(self):
        c = reward_breakdown(np.zeros(6), np.zeros(2), False, True, 0.)
        self.assertEqual(c["reward"], -3010.)

    def test_invalid_overrides_endpoint(self):
        failed, reached, term, trunc, reason = classify_termination(True, True, 0, False)
        self.assertEqual((failed, reached, term, trunc, reason),
                         (True, False, True, False, "invalid_state"))

    def test_solver_failure_overrides_endpoint(self):
        failed, reached, term, trunc, reason = classify_termination(True, False, 3, False)
        self.assertEqual((failed, reached, term, trunc, reason),
                         (True, False, True, False, "solver_fail_3x"))

    def test_success_not_penalized(self):
        status = classify_termination(True, False, 0, False)
        self.assertEqual(status, (False, True, True, False, "reached_end"))
        c = reward_breakdown(np.zeros(6), np.zeros(2), True, status[0], 1.)
        self.assertEqual(c["terminal_failure_cost"], 0.)

    def test_timeout_bootstraps_and_no_terminal_penalty(self):
        status = classify_termination(False, False, 0, True)
        self.assertEqual(status, (False, False, False, True, "time_limit_truncation"))
        c = reward_breakdown(np.zeros(6), np.zeros(2), True, status[0], .5)
        self.assertEqual(c["terminal_failure_cost"], 0.)
        target, h = n_step_targets(np.array([-1.]), np.array([-5.]),
                                   np.array([float(status[2])]), 16)
        self.assertEqual(h.tolist(), [1])
        self.assertAlmostEqual(target[0], -1. - GAMMA*5.)

    def test_termination_disables_bootstrap(self):
        target, h = n_step_targets(np.array([-3010.]), np.array([-5.]),
                                   np.array([1.]), 16)
        self.assertEqual(h.tolist(), [1])
        self.assertEqual(target.tolist(), [-3010.])

    def test_reward_version(self):
        self.assertEqual(REWARD_SCHEME, "fixed_eval_progress_failure_v1")


if __name__ == "__main__":
    unittest.main()
