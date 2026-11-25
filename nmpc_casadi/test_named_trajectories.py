"""Unit tests for the 8m trajectory imports and train/evaluation selector."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trajectory_selection import NEW_TRACKS, TRACK_SHA256, resolve_named_track
from cad_eval_reference import prepare_cad_evaluation_reference
from reference_manager import ReferenceManager


class NamedTrajectoryTests(unittest.TestCase):
    def test_files_and_state_consistency(self):
        for name in NEW_TRACKS:
            with self.subTest(name=name):
                path,digest = resolve_named_track(name)
                self.assertEqual(digest,TRACK_SHA256[name])
                d = pd.read_csv(path)
                self.assertEqual(len(d),801)
                self.assertEqual(float(d.s_m.iloc[0]),0.0)
                self.assertAlmostEqual(float(d.s_m.iloc[-1]),8.0,places=9)
                self.assertTrue(np.all(np.diff(d.s_m.to_numpy())>0))
                self.assertTrue(np.all(np.isfinite(d.to_numpy(dtype=float))))
                np.testing.assert_allclose(d.r_ref_radps,d.vx_ref_mps*d.kappa_1_per_m,rtol=1e-9,atol=1e-9)
                self.assertLessEqual(float(np.max(np.abs(np.arctan(.2515*d.kappa_1_per_m)))),0.55)
                self.assertTrue((d.vx_ref_mps == .1).all())
                self.assertTrue((d.vy_ref_mps == 0).all())

    def test_staging_and_nmpc_reference(self):
        for name in NEW_TRACKS:
            with self.subTest(name=name):
                path,_ = resolve_named_track(name)
                with tempfile.TemporaryDirectory() as temp:
                    info,staged = prepare_cad_evaluation_reference(
                        path,Path(temp),.2,min_speed_mps=.05,max_speed_mps=.1,
                        wheelbase_m=.2515,steering_limit_rad=.55)
                    self.assertAlmostEqual(info['cad_s_end'],8.0)
                    rm = ReferenceManager(staged["tail_file"],stop_at_end=False)
                    horizon = rm.build_horizon_from_s(1.0,.071,10)
                    self.assertEqual(horizon["xref"].shape,(11,6))
                    self.assertTrue(np.isfinite(horizon["xref"]).all())
                    self.assertTrue(np.all(np.diff(horizon["s_horizon"])>=0))


if __name__=="__main__":
    unittest.main()
