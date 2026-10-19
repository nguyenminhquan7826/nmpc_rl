"""Frozen-vs-Online unit checks (run within nmpc_casadi)."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import adapt_nmpc as adapt
import rl_nmpc as rl


class DummyEnv:
    def reset(self, ey, heading, plant):
        self.k = 0
        self.logs = []
        self.applied = []
        return np.zeros(12), np.zeros(6)

    def step(self, weights):
        self.applied.append(tuple(np.asarray(w).copy() for w in weights))
        self.logs.append({'solve_ms': 0.1, 't_s': self.k * rl.b.Ts})
        self.k += 1
        return np.zeros(12), np.zeros(6), -1.0, self.k == 3, self.k == 3


class FrozenTests(unittest.TestCase):
    def test_selected_controllers(self):
        self.assertEqual(len(adapt.active_controllers()), 3)
        self.assertEqual(len(adapt.active_controllers(True)), 4)

    def test_frozen_cannot_update_network(self):
        agent = rl.Agent(seed=123)
        before = adapt.snapshot_network_state(agent)
        independently_seeded = rl.Agent(seed=123)
        expected = [independently_seeded.propose(np.zeros(12), True)[0]
                    for _ in range(3)]
        env = DummyEnv()
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output=Path(directory), payload_start=30., payload_ramp_s=10.,
                road_start=30., road_end=60., road_transition_s=5.,
                payload_fraction=.1, cornering_drop=.2, rolling_increase=.25)
            sc = dict(name='nominal', ey=.05, heading=0., profile='nominal')
            with patch.object(agent, 'learn', side_effect=RuntimeError('unexpected learning')):
                with patch.object(adapt.rl, 'metrics', return_value={}):
                    m = adapt.run_ac_nmpc_frozen(env, agent, args, sc)
            self.assertEqual(m['updates'], 0)
            self.assertTrue(m['frozen_network_unchanged'])
            self.assertTrue(adapt.network_state_unchanged(agent, before))
            log = pd.read_csv(Path(directory) / 'nominal_ac_nmpc_frozen.csv')
            self.assertEqual(len(log), 3)
            for generated, reference in zip(env.applied, expected):
                for x, y in zip(generated, reference):
                    np.testing.assert_array_equal(x, y)
            self.assertTrue((log['learn_ms'] == 0).all())

    def test_deterministic_frozen_uses_same_mean_each_step(self):
        agent = rl.Agent(seed=91)
        expected = rl.Agent(seed=91).propose(np.zeros(12), False)[0]
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output=Path(directory), payload_start=30., payload_ramp_s=10.,
                road_start=30., road_end=60., road_transition_s=5.,
                payload_fraction=.1, cornering_drop=.2, rolling_increase=.25)
            sc = dict(name='nominal', ey=.05, heading=0., profile='nominal')
            env = DummyEnv()
            with patch.object(adapt.rl, 'metrics', return_value={}):
                result = adapt.run_ac_nmpc_frozen(
                    env, agent, args, sc, deterministic=True)
            self.assertEqual(result['sampling_mode'], 'deterministic_mean')
            self.assertEqual(result['updates'], 0)
            for used in env.applied:
                for x, y in zip(used, expected):
                    np.testing.assert_array_equal(x, y)

    def test_parameter_change_detected(self):
        agent = rl.Agent(seed=0)
        before = adapt.snapshot_network_state(agent)
        agent.critic.m[-1][0] += 1.
        self.assertFalse(adapt.network_state_unchanged(agent, before))

    def test_common_pre_update_sampled_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            f = pd.DataFrame({
                'q_0': [1., 2., 3.], 'r_0': [4., 5., 6.],
                'qf_0': [7., 8., 9.], 'cte_m': [0.1, 0.2, 0.3]})
            a = f.copy()
            a.loc[2, 'q_0'] = 200.
            a.loc[2, 'cte_m'] = 9.
            f.to_csv(Path(directory) / 'nominal_ac_nmpc_frozen.csv', index=False)
            a.to_csv(Path(directory) / 'nominal_ac_nmpc_online.csv', index=False)
            stats = adapt.initial_sampled_pairing(directory, 'nominal', 2)
            self.assertEqual(stats['pre_update_steps'], 2)
            self.assertTrue(stats['matched_weight_prefix'])
            self.assertTrue(stats['matched_cte_prefix'])
            later = adapt.initial_sampled_pairing(directory, 'nominal', 3)
            self.assertFalse(later['matched_weight_prefix'])

    def test_pairwise_online_isolation(self):
        rows = []
        for controller, cte, steps in [
            ('fixed_nmpc', .07, 0), ('ac_nmpc_frozen', .02, 0),
            ('ac_nmpc_online', .015, 7),
        ]:
            rows.append(dict(scenario='nominal', plant_profile='nominal',
                controller=controller, updates=steps, reached_end=True,
                tracking_feasible=True, mean_cte_post_m=cte,
                max_cte_post_m=2*cte, rmse_heading_post_rad=.05,
                rmse_vx_post_mps=.01, mean_du_delta_sq=.0001,
                solver_success_pct=100., completion_pct=100.,
                return_=-1000*cte, control_step_total_mean_ms=3.,
                control_step_total_p99_ms=6.))
            rows[-1]['return'] = rows[-1].pop('return_')
        df = pd.DataFrame(rows)
        d = adapt.build_pairwise_summary(df)
        self.assertAlmostEqual(
            float(d.iloc[0]['online_vs_frozen_mean_cte_improvement_pct']), 25.)
        l = adapt.build_pairwise_long(df)
        self.assertEqual(len(l), 3)


if __name__ == '__main__':
    unittest.main()
