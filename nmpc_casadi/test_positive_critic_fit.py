"""Tests for positive-no-upper P regression, KKT, and policy separation."""
import unittest
import numpy as np
from audit_positive_critic_fit import positive_no_upper_least_squares, diagnose_gamma
from audit_critic_returns import EXPECTED_CONDITIONS, fit_metrics


class PositiveCriticFitTests(unittest.TestCase):
    def test_exact_positive_physical_coefficients_are_recovered(self):
        rng = np.random.default_rng(1234)
        errors = rng.normal(size=(500,6)) * np.array([.1,.05,.04,.1,.04,.06])
        x = errors**2
        true = np.array([500.,100.,600.,40.,10000.,11000.])
        y = x @ true
        estimated,diagnostics = positive_no_upper_least_squares(x,y)
        self.assertTrue(diagnostics["kkt_pass"])
        self.assertEqual(diagnostics["rank"],6)
        self.assertEqual(len(diagnostics["active_coordinates"]),6)
        np.testing.assert_allclose(estimated,true,rtol=1e-8,atol=1e-6)

    def test_negative_signed_optimum_projects_to_boundary(self):
        rng = np.random.default_rng(119)
        x = np.square(rng.normal(size=(900,6)))
        signed_p = np.array([2.,0.,-5.,3.,8.,1.])
        y = x @ signed_p
        estimated,stat = positive_no_upper_least_squares(x,y)
        self.assertTrue(stat["kkt_pass"])
        self.assertGreaterEqual(float(estimated.min()),0.)
        self.assertAlmostEqual(estimated[2],0.,places=6)
        self.assertGreater(fit_metrics(x,y,estimated)["rmse"],1e-2)

    def test_recovered_positive_fit_can_exceed_screened_upper(self):
        rng = np.random.default_rng(19)
        x = np.square(rng.normal(size=(200,6)))
        p_true=np.array([800.,800.,800.,800.,8000.,8000.])
        estimated,stat=positive_no_upper_least_squares(x,x@p_true)
        self.assertTrue(stat["kkt_pass"])
        np.testing.assert_allclose(estimated,p_true,rtol=1e-8)
        self.assertGreater(float(estimated[-1]),3465.)

    def test_rank_deficient_features_do_not_create_negative_values(self):
        x=np.zeros((30,6))
        x[:,0]=np.linspace(.1,1.,30)
        y=3.*x[:,0]
        p,meta=positive_no_upper_least_squares(x,y)
        self.assertTrue(meta["kkt_pass"])
        self.assertEqual(meta["rank"],1)
        self.assertGreaterEqual(float(np.min(p)),0.)
        self.assertAlmostEqual(p[0],3.,places=8)
        np.testing.assert_allclose(x@p,y)

    def test_separate_policy_fits_expose_pooled_policy_mixture(self):
        rng=np.random.default_rng(7)
        p_a=np.array([100.,80.,200.,70.,800.,1000.])
        p_b=np.array([100.,80.,200.,70.,3000.,4000.])
        records=[]
        for label,true_p in (("A",p_a),("B",p_b)):
            for ey,heading in EXPECTED_CONDITIONS:
                # Construct rewards so gamma-discounted costs equal
                # x@p policy values at every row. With final state
                # exactly at zero, every reward <=0 on this test.
                x=np.square(rng.normal(size=(70,6))*.05)
                x[-1]=0.
                cost=x@true_p
                gamma=.9
                reward=-(cost-np.r_[gamma*cost[1:],0.])
                # Nonmonotone synthetic x can cause positive r; the
                # discounted-cost routine still handles this identity.
                records.append(dict(label=label,ey_m=ey,
                                    heading_deg=heading,features=x,
                                    reward=reward,n_steps=len(reward)))
        r=diagnose_gamma(.9,records)
        self.assertEqual(len(r["per_fixed_policy"]),2)
        self.assertTrue(r["pooled_nnls_numerical_check"]["kkt_pass"])
        self.assertLess(r["mean_policy_nnls_rmse"],1e-5)
        self.assertGreater(r["mean_policy_pooled_rmse"],r["mean_policy_nnls_rmse"]+0.001)
        self.assertTrue(all(z["nnls_kkt_pass"] for z in r["per_fixed_policy"]))

    def test_reject_invalid_dimensions_and_nonfinite(self):
        with self.assertRaises(ValueError):
            positive_no_upper_least_squares(np.ones((2,6)),np.ones(2))
        with self.assertRaises(ValueError):
            positive_no_upper_least_squares(np.ones((10,6)),np.ones(9))
        bad=np.ones((10,6))
        bad[0,0]=np.nan
        with self.assertRaises(ValueError):
            positive_no_upper_least_squares(bad,np.ones(10))


if __name__=="__main__":
    unittest.main()
