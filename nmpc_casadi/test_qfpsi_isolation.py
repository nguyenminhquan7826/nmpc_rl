"""Qf_psi(s) function-hold tests. Run: python -m unittest -v test_qfpsi_isolation"""
import unittest

import numpy as np

import rl_nmpc as rl
import adapt_nmpc as adapt
from ac_ablation import learn_selected_component, snapshot_net, net_parameter_change_norm
from qfpsi_isolation import (
    QFPSI_INDEX, install_pretrained_qfpsi_function, verify_frozen_qfpsi,
)


def synthetic_rows(seed=42, n=16):
    gen = np.random.default_rng(seed)
    result = []
    for j in range(n):
        x = gen.normal(0, .18, 12)
        e = gen.normal(0, .06, 6)
        z = gen.normal(0, .2, 3)
        xn = x + gen.normal(0, .015, 12)
        en = e + gen.normal(0, .01, 6)
        result.append((x, e, z, -float(j + 1), xn, en, 0.))
    return result


class QfPsiFunctionalIsolationTests(unittest.TestCase):
    def test_optional_roster_and_mode_guard(self):
        self.assertEqual(
            len(adapt.active_controllers(include_ablation=True)), 5,
        )
        self.assertEqual(
            len(adapt.active_controllers(
                include_ablation=True, isolate_qfpsi=True,
            )), 7,
        )
        with self.assertRaisesRegex(ValueError, 'requires'):
            adapt.active_controllers(isolate_qfpsi=True)
        with self.assertRaisesRegex(ValueError, 'only supported'):
            adapt.run_ac_nmpc_online(
                None, rl.Agent(seed=3), None, None,
                train_component='actor',
                controller_label='ac_nmpc_actor_only',
                hold_qfpsi=True,
            )

    def test_pretrained_qfpsi_is_immutable_under_critic_updates(self):
        agent = rl.Agent(seed=7, actor_lr=.001, critic_lr=.001)
        rng = np.random.default_rng(50)
        states = rng.normal(0, .15, (8, 12))
        original_terms = agent.critic_terms
        reference = original_terms(states)[0][:, QFPSI_INDEX].copy()
        previous = snapshot_net(agent.critic)
        old_method, reference_fn = install_pretrained_qfpsi_function(agent)
        self.assertEqual(old_method.__func__, original_terms.__func__)
        try:
            for _ in range(6):
                result = learn_selected_component(
                    agent, synthetic_rows(), mode='critic',
                )
                self.assertGreater(result['critic_parameter_delta'], 0.)
                held = agent.critic_terms(states)
                np.testing.assert_array_equal(
                    held[0][:, QFPSI_INDEX], reference,
                )
                np.testing.assert_array_equal(
                    held[1][:, QFPSI_INDEX], np.zeros(len(states)),
                )
                np.testing.assert_allclose(
                    reference_fn(states), rl.F_BASE[QFPSI_INDEX]*reference,
                    rtol=0, atol=0,
                )
                # Controller weights must use held terminal heading penalty.
                weights, _ = agent.propose(states[0], explore=False)
                verify_frozen_qfpsi(
                    weights, float(reference_fn(states[0])),
                )
            self.assertGreater(
                net_parameter_change_norm(agent.critic, previous), 0.,
            )
        finally:
            agent.critic_terms = old_method

    def test_other_critic_outputs_still_adapt(self):
        agent = rl.Agent(seed=11, actor_lr=.001, critic_lr=.001)
        states = np.random.default_rng(1).normal(0, .2, (12, 12))
        original_terms, reference_fn = install_pretrained_qfpsi_function(agent)
        before = agent.critic_terms(states)[0].copy()
        try:
            for _ in range(6):
                learn_selected_component(agent, synthetic_rows(seed=8), 'critic')
            after = agent.critic_terms(states)[0]
            np.testing.assert_array_equal(after[:, 2], before[:, 2])
            self.assertTrue(np.any(
                np.abs(np.delete(after, 2, axis=1) -
                       np.delete(before, 2, axis=1)) > 1e-8
            ))
        finally:
            agent.critic_terms = original_terms

    def test_held_mask_preserves_state_dependent_function(self):
        a = rl.Agent(seed=13)
        gen = np.random.default_rng(55)
        x = gen.normal(0, .12, (5, 12))
        start = a.critic_terms(x)[0][:, 2].copy()
        old, reference = install_pretrained_qfpsi_function(a)
        try:
            np.testing.assert_array_equal(a.critic_terms(x)[0][:, 2], start)
            assert np.isfinite(reference(x)).all()
            with self.assertRaisesRegex(RuntimeError, 'not held'):
                verify_frozen_qfpsi(
                    a.propose(x[0], False)[0],
                    float(reference(x[0])) + 1.,
                )
        finally:
            a.critic_terms = old


if __name__ == '__main__':
    unittest.main()
