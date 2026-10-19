"""Checks that AutoCAD Evaluation XREF never overwrites Training references.

Run from nmpc_casadi:
    python -m unittest -v test_cad_eval_reference
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import adapt_nmpc as adapt
import rl_nmpc as rl
from cad_eval_reference import prepare_cad_evaluation_reference
from heldout_reference import sha256_file


def make_variable_speed_reference(path):
    s = np.linspace(0., 5., 201)
    vx = .08 + .02*np.sin(2*np.pi*s/5.)**2
    t = np.r_[0., np.cumsum(np.diff(s) / (.5*(vx[:-1] + vx[1:])))]
    data = pd.DataFrame({
        'index': np.arange(len(s)), 's_m': s, 't_ref_s': t,
        'vx_ref_mps': vx, 'vy_ref_mps': np.zeros_like(s),
        'psi_ref_rad': np.zeros_like(s), 'r_ref_radps': np.zeros_like(s),
        'X_ref_m': s, 'Y_ref_m': np.zeros_like(s),
        'kappa_1_per_m': np.zeros_like(s),
    })
    data.to_csv(path, index=False)
    return data


class CadEvaluationReferenceTests(unittest.TestCase):
    def test_custom_cad_preserves_velocity_and_source_sha(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / 'cad_new.csv'
            make_variable_speed_reference(src)
            original_sha = sha256_file(src)
            info, meta = prepare_cad_evaluation_reference(
                src, td / 'output', .20
            )
            self.assertEqual(sha256_file(src), original_sha)
            self.assertEqual(meta['reference_sha256'], original_sha)
            self.assertGreater(info['tail_s_end'], info['cad_s_end'])
            src_data = pd.read_csv(src)
            reference = pd.read_csv(meta['tail_file'])
            np.testing.assert_array_equal(
                reference.vx_ref_mps.values[:len(src_data)],
                src_data.vx_ref_mps.values,
            )
            self.assertAlmostEqual(
                reference.vx_ref_mps.iloc[-1],
                src_data.vx_ref_mps.iloc[-1],
            )
            self.assertGreater(meta['max_reference_speed_mps'],
                               meta['min_reference_speed_mps'])

    def test_evaluation_runtime_switch_is_isolated_from_training(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / 'cad_new.csv'
            expected = make_variable_speed_reference(src)
            original_runtime = (rl.b.TAIL_FILE, rl.b.REF_INFO, rl.b.CAD_S_END)
            training_hash = sha256_file(adapt.TRAINING_REF_SOURCE)
            try:
                meta = adapt.install_evaluation_track(
                    td / 'output', 'heldout_s_bend', track_csv=src,
                )
                self.assertEqual(meta['id'], 'cad_eval_external')
                self.assertTrue(meta['profile_preserved_from_source'])
                self.assertTrue(meta['held_out'])
                self.assertAlmostEqual(rl.b.CAD_S_END, 5.)
                rm = rl.b.make_reference_manager()
                self.assertAlmostEqual(
                    rm.interpolate_reference(1.)[0],
                    float(np.interp(
                        1., expected.s_m.to_numpy(float),
                        expected.vx_ref_mps.to_numpy(float),
                    )),
                )
                self.assertEqual(sha256_file(adapt.TRAINING_REF_SOURCE),
                                 training_hash)
            finally:
                rl.b.TAIL_FILE, rl.b.REF_INFO, rl.b.CAD_S_END = original_runtime

    def test_reject_training_reference_as_heldout(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, 'training CAD source'):
                adapt.install_evaluation_track(
                    Path(td), 'heldout_s_bend',
                    track_csv=adapt.TRAINING_REF_SOURCE,
                )

    def test_reject_invalid_speed_and_timing(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / 'test.csv'
            base = make_variable_speed_reference(src)
            modified = base.copy()
            modified.loc[3, 'vx_ref_mps'] = .14
            modified.to_csv(src, index=False)
            with self.assertRaisesRegex(ValueError, 'outside NMPC/test'):
                prepare_cad_evaluation_reference(src, td / 'output1', .20)
            modified = base.copy()
            modified.loc[3, 't_ref_s'] += 1.
            modified.to_csv(src, index=False)
            with self.assertRaisesRegex(ValueError, 'inconsistent'):
                prepare_cad_evaluation_reference(src, td / 'output2', .20)


if __name__ == '__main__':
    unittest.main()
