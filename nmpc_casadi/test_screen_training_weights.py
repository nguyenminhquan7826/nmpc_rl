"""Pure unit tests for approved start and exploratory TRAIN-only screening."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from screen_training_weights import (
    verify_train_reference, initial_conditions, screen_one,
)
from training_initialization import (
    LEARNING_INITIAL, GOOD_NMPC, TRAIN_REFERENCE_SHA256,
    representative_weights, screening_box,
)


class ScreeningSpecTests(unittest.TestCase):
    def test_learning_initial_matches_completed_v3_start(self):
        q, r, p = LEARNING_INITIAL.arrays()
        np.testing.assert_allclose(q, [12.,3.,50.,6.,100.75,100.75])
        np.testing.assert_allclose(r, [5.,10.])
        np.testing.assert_allclose(p, [120.,30.,500.,60.,1007.5,1007.5])

    def test_strong_comparator_matches_completed_v3(self):
        q,r,p = GOOD_NMPC.arrays()
        np.testing.assert_allclose(q,[12.,3.,15.,6.,183.75,192.5])
        np.testing.assert_allclose(r,[.6,16.])
        np.testing.assert_allclose(p,20*q)
        # Comparators do not alias each other.
        a=LEARNING_INITIAL.arrays()
        q[0]=999
        self.assertEqual(GOOD_NMPC.q[0],12.)
        self.assertEqual(a[0][0],12.)

    def test_candidate_box_contains_both_in_strict_interior(self):
        low, high=screening_box(.40)
        initial=LEARNING_INITIAL.arrays()
        fixed=GOOD_NMPC.arrays()
        self.assertEqual([len(x) for x in low],[6,2,6])
        np.testing.assert_allclose(low[0],[7.2,1.8,9.,3.6,60.45,60.45])
        np.testing.assert_allclose(high[0],[16.8,4.2,70.,8.4,257.25,269.5])
        np.testing.assert_allclose(low[1],[.36,6.])
        np.testing.assert_allclose(high[1],[7.,22.4])
        np.testing.assert_allclose(low[2],[72.,18.,180.,36.,604.5,604.5])
        np.testing.assert_allclose(high[2],[336.,84.,700.,168.,5145.,5390.])
        for bounds_l,bounds_u,a,b in zip(low,high,initial,fixed):
            self.assertTrue(np.all(bounds_l<a))
            self.assertTrue(np.all(a<bounds_u))
            self.assertTrue(np.all(bounds_l<b))
            self.assertTrue(np.all(b<bounds_u))

    def test_requires_explicit_valid_screening_margin(self):
        for margin in [-.1,0.,1.,float('nan'),float('inf')]:
            with self.subTest(margin=margin),self.assertRaises(ValueError):
                screening_box(margin)

    def test_quick_and_full_have_expected_coverage(self):
        quick=representative_weights(.20,"quick")
        full=representative_weights(.20,"full")
        self.assertEqual(len(quick),3)
        self.assertEqual(len(full),9)
        self.assertEqual(len(initial_conditions("quick")),1)
        self.assertEqual(len(initial_conditions("full")),3)
        self.assertEqual(set(quick).issubset(set(full)),True)
        for w in full.values():
            self.assertEqual([x.shape for x in w],[(6,),(2,),(6,)])
            self.assertTrue(all(np.all(x>0) for x in w))

    def test_unrecognized_reference_sha_refused(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"train_xref.csv"
            path.write_text("reference,not,the,pinned,training,file\n")
            with self.assertRaisesRegex(ValueError,"SHA256 mismatch"):
                verify_train_reference(path)
        self.assertEqual(len(TRAIN_REFERENCE_SHA256),64)

    def test_simulation_failure_is_recorded_not_counted_as_pass(self):
        with tempfile.TemporaryDirectory() as td:
            weights=LEARNING_INITIAL.arrays()
            class FakeProcess:
                returncode=1
                stdout="test failure"
                stderr="solver crash"
            with patch("screen_training_weights.subprocess.run",
                       return_value=FakeProcess()) as run:
                result=screen_one(Path(td)/"train_xref.csv",Path(td),
                                  "fail_test",weights,.05,0.,150.)
            self.assertEqual(result["status"],"PROCESS_FAILED")
            self.assertEqual(result["returncode"],1)
            call_args=run.call_args.args[0]
            self.assertIn("--no-plots",call_args)
            self.assertIn("--qf",call_args)

    def test_simulation_success_is_only_empirical_label(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            weights=LEARNING_INITIAL.arrays()
            class FakeProcess:
                returncode=0
                stdout=""
                stderr=""
            def fake_run(args,**kwargs):
                case_dir=Path(args[args.index("--output")+1])
                case_dir.mkdir()
                (case_dir/"summary.json").write_text(json.dumps({
                    "reached_end":True,"solver_success_pct":100.,
                    "solver_aborted":False,"tracking_feasible":True,
                    "mean_cte_post_m":.0200,
                }))
                (case_dir/"config.json").write_text(json.dumps({
                    "reference_sha256":TRAIN_REFERENCE_SHA256
                }))
                return FakeProcess()
            with patch("screen_training_weights.subprocess.run",side_effect=fake_run):
                result=screen_one(root/"train_xref.csv",root,
                                  "pass_test",weights,.05,0.,150.)
            self.assertEqual(result["status"],"EMPIRICAL_PASS")
            self.assertAlmostEqual(result["mean_cte_post_m"],.0200)


if __name__ == "__main__":
    unittest.main()
