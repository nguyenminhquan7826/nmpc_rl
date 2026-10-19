"""Regression: distinct Evaluation sampling seeds must survive checkpoint load.

Run from nmpc_casadi:
    python -m unittest -v test_sampling_seed
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import adapt_nmpc as adapt
import rl_nmpc as rl


class EvaluationSamplingSeedTests(unittest.TestCase):
    def _checkpoint(self, folder):
        path = Path(folder) / 'sampled_checkpoint.npz'
        agent = rl.Agent(seed=209, actor_lr=.001, critic_lr=.001)
        state = np.zeros(12)
        for _ in range(7):
            agent.propose(state, True)
        agent.save(path)
        with np.load(path, allow_pickle=False) as checkpoint:
            meta = json.loads(str(checkpoint['metadata'].item()))
        return path, meta

    def _args(self, path, sampling_seed=None):
        return SimpleNamespace(
            checkpoint=path, seed=0, sampling_seed=sampling_seed,
            actor_lr=.001, critic_lr=.001,
        )

    def test_default_restores_checkpoint_rng_exactly(self):
        with tempfile.TemporaryDirectory() as folder:
            path, meta = self._checkpoint(folder)
            args = self._args(path)
            agent = adapt.fresh_agent(args, meta)
            direct = rl.Agent(seed=9, actor_lr=.001, critic_lr=.001)
            direct.load(path)
            state = np.zeros(12)
            np.testing.assert_array_equal(
                agent.propose(state, True)[1],
                direct.propose(state, True)[1],
            )
            self.assertTrue(adapt.network_state_unchanged(
                agent, adapt.snapshot_network_state(direct),
            ))

    def test_explicit_seed_reproducible_and_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            path, meta = self._checkpoint(folder)
            old_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            first = adapt.fresh_agent(self._args(path, 5), meta)
            second = adapt.fresh_agent(self._args(path, 5), meta)
            third = adapt.fresh_agent(self._args(path, 6), meta)
            state = np.zeros(12)
            before_first = adapt.snapshot_network_state(first)
            before_third = adapt.snapshot_network_state(third)
            self.assertTrue(adapt.network_state_unchanged(first, before_third))
            weights_first, z_first = first.propose(state, True)
            weights_second, z_second = second.propose(state, True)
            weights_third, z_third = third.propose(state, True)
            np.testing.assert_array_equal(z_first, z_second)
            self.assertFalse(np.array_equal(z_first, z_third))
            for w1, w2 in zip(weights_first, weights_second):
                np.testing.assert_array_equal(w1, w2)
            self.assertTrue(adapt.network_state_unchanged(first, before_first))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), old_hash)

    def test_seed_zero_changes_rng_from_checkpoint_but_not_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            path, meta = self._checkpoint(folder)
            default = adapt.fresh_agent(self._args(path), meta)
            explicit_zero = adapt.fresh_agent(self._args(path, 0), meta)
            self.assertTrue(adapt.network_state_unchanged(
                explicit_zero, adapt.snapshot_network_state(default),
            ))
            self.assertFalse(np.array_equal(
                default.propose(np.zeros(12), True)[1],
                explicit_zero.propose(np.zeros(12), True)[1],
            ))


if __name__ == '__main__':
    unittest.main()
