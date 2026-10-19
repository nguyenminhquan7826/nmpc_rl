"""Out-of-distribution track validation; no solver run needed.

Run from nmpc_casadi:
    python -m unittest -v test_adapt_frozen test_heldout_reference
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import adapt_nmpc as adapt
import rl_nmpc as rl
from heldout_reference import (
    HELDOUT_ID, compare_geometry, describe_geometry,
    generate_heldout_s_bend, sha256_file,
)


class HeldoutGeometryTests(unittest.TestCase):
    def test_new_s_bend_is_geometrically_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'heldout.csv'
            data = generate_heldout_s_bend(
                target, adapt.TRAINING_LENGTH_M,
                speed_mps=rl.b.TEST_SPEED_MPS)
            self.assertTrue(target.is_file())
            self.assertAlmostEqual(
                data.s_m.iloc[-1], adapt.TRAINING_LENGTH_M, places=9)
            self.assertGreater(len(data), 150)
            self.assertTrue((data.s_m.diff().dropna() > 0).all())
            self.assertTrue(np.allclose(
                data.r_ref_radps,
                data.vx_ref_mps * data.kappa_1_per_m,
                rtol=1e-10, atol=1e-10
            ))
            self.assertGreater(float(data.kappa_1_per_m.max()), .9)
            self.assertLess(float(data.kappa_1_per_m.min()), -.9)
            self.assertLess(float(np.max(
                np.abs(data.kappa_1_per_m))), 1.55)
            diff = compare_geometry(adapt.TRAINING_REF_SOURCE, data)
            self.assertGreater(
                diff['rms_xy_separation_from_training_m'], .5)
            self.assertGreater(
                diff['rms_curvature_difference_1_per_m'], .1)
            self.assertFalse(diff['exact_same_waypoints'])

            # Repeated runs must generate identical CSV bytes.
            other = Path(directory) / 'repeat.csv'
            generate_heldout_s_bend(
                other, adapt.TRAINING_LENGTH_M,
                speed_mps=rl.b.TEST_SPEED_MPS)
            self.assertEqual(sha256_file(target), sha256_file(other))

    def test_reference_manager_uses_new_path_and_virtual_tail(self):
        original = (rl.b.TAIL_FILE, rl.b.REF_INFO, rl.b.CAD_S_END)
        with tempfile.TemporaryDirectory() as directory:
            try:
                meta = adapt.install_evaluation_track(
                    Path(directory), 'heldout_s_bend')
                self.assertTrue(meta['held_out'])
                self.assertEqual(meta['id'], HELDOUT_ID)
                self.assertTrue(meta['not_used_for_training_or_checkpoint_selection'])
                self.assertNotEqual(
                    meta['reference_sha256'],
                    meta['training_reference_sha256'])
                self.assertAlmostEqual(
                    rl.b.CAD_S_END, adapt.TRAINING_LENGTH_M, places=8)
                self.assertGreater(
                    rl.b.REF_INFO['tail_s_end'], rl.b.CAD_S_END)
                manager = rl.b.make_reference_manager()
                s = manager.s
                self.assertGreater(s[-1], rl.b.CAD_S_END)
                self.assertTrue(np.all(np.diff(s) > 0.))
                manifest = json.loads(
                    (Path(directory) / 'reference_manifest.json')
                    .read_text(encoding='utf-8')
                )
                self.assertEqual(manifest['id'], HELDOUT_ID)
                self.assertAlmostEqual(
                    float(manager.interpolate_reference(.1)[0]),
                    rl.b.TEST_SPEED_MPS, places=8)
            finally:
                rl.b.TAIL_FILE, rl.b.REF_INFO, rl.b.CAD_S_END = original

    def test_replay_must_be_explicit(self):
        original = (rl.b.TAIL_FILE, rl.b.REF_INFO, rl.b.CAD_S_END)
        with tempfile.TemporaryDirectory() as directory:
            try:
                meta = adapt.install_evaluation_track(
                    Path(directory), 'training')
                self.assertFalse(meta['held_out'])
                self.assertEqual(meta['id'], 'training_original')
            finally:
                rl.b.TAIL_FILE, rl.b.REF_INFO, rl.b.CAD_S_END = original

    def test_reject_nonphysical_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'x.csv'
            for options in [
                dict(length_m=0.0),
                dict(length_m=7.0, speed_mps=-.1),
                dict(length_m=7.0, ds_target_m=.2),
            ]:
                with self.assertRaises(ValueError):
                    generate_heldout_s_bend(path, **options)


if __name__ == '__main__':
    unittest.main()
