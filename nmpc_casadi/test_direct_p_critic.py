"""Six physical terminal weights, no neural Critic; paper TD(0) tests.

Run: python -m unittest -v test_direct_p_critic
"""
import unittest

import numpy as np

import baseline as b
from direct_p_critic import SixParameterTDCritic, error_from_state
from critic_nmpc_bridge import actor_critic_nmpc_parameters
from nmpc_casadi import build_nmpc_solver, pack_parameters, tracking_error


class DirectPCriticTests(unittest.TestCase):
    def new_critic(self, p=None):
        if p is None:
            p = np.ones(6)
        return SixParameterTDCritic(p, np.full(6, .25), np.full(6, 3.))

    def test_exactly_six_trainable_entries_no_nn(self):
        c = self.new_critic()
        self.assertEqual(c.p.shape, (6,))
        self.assertEqual(c.terminal_matrix().shape, (6, 6))
        self.assertEqual(c.__dict__.keys(), {"_p", "lower", "upper"})
        mutated_copy = c.p
        mutated_copy[0] = 999.
        self.assertEqual(c.p[0], 1.)

    def test_negative_td_increases_visited_P_correct_sign(self):
        c = self.new_critic()
        e = np.array([1., 0., 0., 0., 0., 0.])
        result = c.td_step(e, np.zeros(6), -2., .99, .1)
        self.assertAlmostEqual(result.value_current, -1.)
        self.assertAlmostEqual(result.td_error, -1.)
        self.assertAlmostEqual(result.loss, .5)
        np.testing.assert_allclose(result.semi_gradient, [-1., 0, 0, 0, 0, 0])
        np.testing.assert_allclose(c.p, [1.1, 1., 1., 1., 1., 1.])

    def test_zero_error_leaves_p_unchanged_even_with_td(self):
        c = self.new_critic()
        z = np.zeros(6)
        res = c.td_step(z, z, -4., .99, .1)
        self.assertAlmostEqual(res.td_error, -4.)
        np.testing.assert_allclose(c.p, np.ones(6))

    def test_semi_gradient_equals_finite_difference_of_frozen_target(self):
        c = self.new_critic()
        e = np.array([.1, -.2, .03, .4, -.05, .07])
        en = np.array([.2, .1, -.01, .2, .07, -.06])
        gamma, reward = .93, -.75
        old_p = c.p.copy()
        next_value = c.value(en)
        target = reward+gamma*next_value
        delta = target-c.value(e)
        analytic = delta*e**2
        def frozen_loss(p):
            value = -np.dot(p, e**2)
            return .5*(target-value)**2
        eps = 1e-5
        finite = []
        for i in range(6):
            pos, neg = old_p.copy(), old_p.copy()
            pos[i] += eps
            neg[i] -= eps
            finite.append((frozen_loss(pos)-frozen_loss(neg))/(2*eps))
        np.testing.assert_allclose(analytic, finite, atol=1e-10, rtol=1e-6)

    def test_true_terminal_drops_bootstrap_but_timeout_does_not(self):
        e, en = np.ones(6), np.full(6, .5)
        a, bcritic = self.new_critic(), self.new_critic()
        ra = a.td_step(e, en, -1., .99, .01, terminated=True)
        rb = bcritic.td_step(e, en, -1., .99, .01, terminated=False)
        self.assertEqual(ra.value_next, 0.)
        self.assertAlmostEqual(rb.value_next, -1.5)
        self.assertNotEqual(ra.td_error, rb.td_error)

    def test_projected_bounds_and_initial_validation(self):
        c = SixParameterTDCritic(np.ones(6), np.full(6,.5), np.full(6,2.))
        c.td_step(np.ones(6), np.zeros(6), -100., .99, 1.)
        np.testing.assert_allclose(c.p, 2*np.ones(6))
        c.td_step(np.ones(6), np.zeros(6), 100., .99, 1.)
        np.testing.assert_allclose(c.p, .5*np.ones(6))
        with self.assertRaises(ValueError):
            SixParameterTDCritic(np.ones(6), np.zeros(6), np.ones(6)*3)
        with self.assertRaises(ValueError):
            SixParameterTDCritic(np.ones(6)*4, np.ones(6), np.ones(6)*3)

    def test_heading_error_matches_nmPC_wrap(self):
        x = np.array([.1, 0., -np.pi+.01, .01, 1., 2.])
        xref = np.array([.1, 0., np.pi-.01, .01, 1., 2.])
        e = error_from_state(x, xref)
        np.testing.assert_allclose(
            e, np.asarray(tracking_error(x, xref)).reshape(-1), atol=1e-14)
        self.assertAlmostEqual(e[2], .02)

    def test_critic_p_is_terminal_qf_in_solver_parameters(self):
        q, r, qf = b.get_robust_controller_weights()
        p0 = np.diag(qf)
        critic = SixParameterTDCritic(p0, p0*.5, p0*2.)
        problem = build_nmpc_solver(b.Fd_controller, b.N, q, r, qf, b.cfg,
                                    adaptive_cost_weights=True)
        x0 = np.array([.1,0.,0.,0.,0.,0.])
        prev = np.zeros(2)
        xr = np.zeros((b.N+1, 6))
        par = actor_critic_nmpc_parameters(
            x0, prev, xr, b.p_nominal, np.diag(q), np.diag(r), critic)
        self.assertEqual(par.size, problem["n_parameter"])
        np.testing.assert_allclose(par[-14:-8], np.diag(q))
        np.testing.assert_allclose(par[-8:-6], np.diag(r))
        np.testing.assert_allclose(par[-6:], p0)
        critic.td_step(np.array([.1,0.,0.,0.,0.,0.]),np.zeros(6),-1.,.99,.01)
        par2 = actor_critic_nmpc_parameters(
            x0, prev, xr, b.p_nominal, np.diag(q), np.diag(r), critic)
        # With the baseline P0=180, e_vx=0.1 and r=-1, delta>0;
        # the correct semi-gradient update lowers p_vx, not raises it.
        self.assertLess(par2[-6], par[-6])

    def test_pack_default_fixed_interface_unchanged(self):
        q,r,qf = b.get_robust_controller_weights()
        fixed = build_nmpc_solver(b.Fd_controller,b.N,q,r,qf,b.cfg)
        adaptive = build_nmpc_solver(
            b.Fd_controller,b.N,q,r,qf,b.cfg,adaptive_cost_weights=True)
        base = pack_parameters(np.zeros(6), np.zeros(2), np.zeros((b.N+1,6)), b.p_nominal)
        self.assertEqual(base.size,fixed["n_parameter"])
        self.assertEqual(base.size+14, adaptive["n_parameter"])
        with self.assertRaises(ValueError):
            pack_parameters(np.zeros(6), np.zeros(2), np.zeros((b.N+1,6)),
                            b.p_nominal, weights=(np.zeros(6),np.ones(2),np.ones(6)))


if __name__ == "__main__":
    unittest.main()
