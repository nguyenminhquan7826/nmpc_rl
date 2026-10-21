"""Independent numerical tests for return-to-go and Direct-P fit diagnostics."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from audit_critic_returns import (
    EXPECTED_CONDITIONS,
    observed_discounted_costs,
    error_features_and_reward,
    fitted_box_least_squares,
    signed_unrestricted_least_squares,
    fit_metrics,
    evaluate_discount,
    load_full_train_screening,
)
from training_initialization import (
    LEARNING_INITIAL, TRAIN_REFERENCE_SHA256, screening_box,
)


class CriticReturnAuditTests(unittest.TestCase):
    def test_discounted_cost_to_go_includes_final_observed_reward(self):
        reward=np.array([-1.,-2.,-3.])
        expected=np.array([1.+.5*2.+.25*3.,2.+.5*3.,3.])
        np.testing.assert_allclose(observed_discounted_costs(reward,.5),
                                   expected,atol=1e-14)
        np.testing.assert_allclose(observed_discounted_costs(reward,1.),
                                   [6.,5.,3.])
        with self.assertRaises(ValueError):
            observed_discounted_costs(reward,0.)
        with self.assertRaises(ValueError):
            observed_discounted_costs(np.array([]),.99)

    def test_reward_is_fixed_paper_eq4_not_adaptive_qr(self):
        df=pd.DataFrame({
            "e_vx":[0.], "e_vy":[0.],"e_psi":[2*np.pi+.1],
            "e_r":[0.],"e_X":[.02],"e_Y":[-.05],
            "du_a":[.03],"du_delta":[-.04],
        })
        feats,reward=error_features_and_reward(df)
        self.assertAlmostEqual(feats[0,2],.01)
        self.assertAlmostEqual(feats[0,5],.0025)
        expected=-(100*.01+600*.02**2+600*.05**2+
                   50*.03**2+100*.04**2)
        self.assertAlmostEqual(reward[0],expected)

    def test_full_rank_fit_recovers_true_physical_p(self):
        rng=np.random.default_rng(7)
        e=rng.normal(size=(600,6))*np.array([.1,.05,.2,.08,.04,.08])
        x=e*e
        true=np.array([110.,28.,80.,70.,1000.,1300.])
        target=x@true
        lower,upper=screening_box(.2)
        estimate,diag=fitted_box_least_squares(
            x,target,lower[2],upper[2],LEARNING_INITIAL.arrays()[2])
        self.assertTrue(diag["converged"])
        np.testing.assert_allclose(estimate,true,rtol=1e-6,atol=1e-4)
        self.assertLess(fit_metrics(x,target,estimate)["rmse"],1e-5)
        signed,rank=signed_unrestricted_least_squares(x,target)
        self.assertEqual(rank,6)
        np.testing.assert_allclose(signed,true,rtol=1e-8,atol=1e-7)

    def test_box_fit_hits_boundary_when_true_p_outside(self):
        rng=np.random.default_rng(42)
        x=np.square(rng.normal(size=(500,6)))
        initial=np.full(6,2.)
        low=np.full(6,1.)
        high=np.full(6,5.)
        true=np.array([10.,2.,3.,4.,2.,3.])
        y=x@true
        p,diagnostics=fitted_box_least_squares(x,y,low,high,initial)
        self.assertTrue(diagnostics["converged"])
        self.assertGreaterEqual(p[0],4.99999)
        self.assertTrue(np.all(p>=low-1e-10))
        self.assertTrue(np.all(p<=high+1e-10))
        self.assertLess(fit_metrics(x,y,p)["rmse"],
                        fit_metrics(x,y,initial)["rmse"])

    def test_rank_deficient_can_be_reported_without_nans(self):
        x=np.zeros((20,6))
        x[:,0]=np.linspace(.01,1.,20)
        y=1.3*x[:,0]
        lower=np.ones(6)
        upper=np.full(6,10.)
        p,diag=fitted_box_least_squares(
            x,y,lower,upper,np.full(6,2.))
        self.assertTrue(diag["converged"])
        self.assertEqual(diag["inactive_columns"],[1,2,3,4,5])
        self.assertAlmostEqual(p[0],1.3)
        self.assertTrue(np.all(np.isfinite(p)))

    def test_holdout_by_training_initial_condition_not_test(self):
        rng=np.random.default_rng(3)
        records=[]
        for c in EXPECTED_CONDITIONS:
            for name in ("a","b"):
                x=np.square(rng.normal(size=(20,6))*.07)
                reward=-np.linspace(.5,.2,20)
                records.append(dict(
                    label=name,ey_m=c[0],heading_deg=c[1],
                    features=x,reward=reward,n_steps=20))
        lo,hi=screening_box(.2)
        d=evaluate_discount(.95,records,lo[2],hi[2],LEARNING_INITIAL.arrays()[2])
        self.assertEqual(d["n_cases"],6)
        self.assertEqual(d["sample_count"],120)
        self.assertEqual(len(d["cases"]),6)
        self.assertEqual(len(d["heldout_within_train_by_initial_condition"]),3)
        self.assertTrue(np.isfinite(d["box_fit_metrics_in_sample"]["rmse"]))
        for fold in d["heldout_within_train_by_initial_condition"]:
            self.assertEqual(len(fold["trained_p"]),6)

    def test_reject_incomplete_or_unpinned_full_screening(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"manifest.json"
            path.write_text(json.dumps({
                "status":"EXPLORATORY_ONLY_NOT_CERTIFIED",
                "profile":"full","train_sha256":"test_hash_or_wrong",
                "total_samples":27,"pass_samples":27,
                "empirical_all_pass":True,
            }))
            with self.assertRaisesRegex(ValueError,"pinned TRAIN"):
                load_full_train_screening(tmp)


if __name__=="__main__":
    unittest.main()
