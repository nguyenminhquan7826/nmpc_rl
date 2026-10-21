"""Checkpoint loading and frozen-mode evaluation unit tests (TRAIN only)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from audit_training_reward import Q_EVAL_CANDIDATE,R_EVAL_CANDIDATE
from direct_p_critic import SixParameterTDCritic
from gaussian_weight_actor import GaussianWeightActor
from evaluate_frozen_actor_critic import (
    load_frozen_agents,metrics_from_rows,frozen_rollout,
)
from train_actor_critic import write_checkpoint
from training_initialization import LEARNING_INITIAL, TRAIN_REFERENCE_SHA256, screening_box


def fresh_checkpoint(path, *, gamma=.9, change=True):
    q,r,p=LEARNING_INITIAL.arrays()
    lo,hi=screening_box(.20)
    actor=GaussianWeightActor(
        np.random.default_rng(0),q,r,lo[0],hi[0],lo[1],hi[1],
        sigma_lower=.02,sigma_initial=.1,sigma_upper=.3)
    critic=SixParameterTDCritic(p,lo[2],hi[2])
    if change:
        actor.params["b_mu"][0]+=1e-4
        critic._p[0]+=1e-4
    metadata={
        "reference_role":"TRAIN_ONLY",
        "train_sha256":TRAIN_REFERENCE_SHA256,
        "gamma":gamma,"Qe_fixed":Q_EVAL_CANDIDATE.tolist(),
        "Re_fixed":R_EVAL_CANDIDATE.tolist(),
        "initial_Q":q.tolist(),"initial_R":r.tolist(),
        "initial_P":p.tolist(),
        "Q_lower":lo[0].tolist(),"Q_upper":hi[0].tolist(),
        "R_lower":lo[1].tolist(),"R_upper":hi[1].tolist(),
        "P_lower":lo[2].tolist(),"P_upper":hi[2].tolist(),
        "sigma_lower":.02,"sigma_initial":.1,
        "sigma_upper":.3,"seed":0,"checkpoint_episode":0}
    write_checkpoint(path,actor,critic,metadata)
    return lo,hi


class FreezeEvaluationTests(unittest.TestCase):
    def test_checkpoint_is_loaded_without_pickle_and_both_agents_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"one.npz"
            lo,hi=fresh_checkpoint(path)
            (a0,c0),(a1,c1),meta=load_frozen_agents(path,lo,hi)
            self.assertEqual(meta["gamma"],.9)
            self.assertEqual(a0.parameter_count,6352)
            self.assertNotEqual(a0.params["b_mu"][0],a1.params["b_mu"][0])
            self.assertGreater(c1.p[0],c0.p[0])

    def test_wrong_gamma_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"old.npz"
            lo,hi=fresh_checkpoint(path,gamma=.99)
            with self.assertRaisesRegex(ValueError,"mismatch"):
                load_frozen_agents(path,lo,hi)

    def test_unchanged_actor_checkpoint_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"unchanged.npz"
            lo,hi=fresh_checkpoint(path,change=False)
            with self.assertRaisesRegex(ValueError,"unchanged"):
                load_frozen_agents(path,lo,hi)

    def test_invalid_critic_outside_screen_box_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"bad.npz"
            lo,hi=fresh_checkpoint(path)
            with np.load(path,allow_pickle=False) as z:
                payload={name:z[name].copy() for name in z.files}
            payload["critic_physical_P"]=hi[2]*2.
            np.savez_compressed(path,**payload)
            with self.assertRaisesRegex(ValueError,"outside"):
                load_frozen_agents(path,lo,hi)

    def test_train_metrics_use_same_pre_step_errors_and_discount(self):
        log=pd.DataFrame({
            "s_progress_m":[0.,1.3,7.95],
            "cross_track_m":[.05,.02,.01],
            "e_psi":[.2,.04,.02],"e_vx":[.05,.01,.01],
            "solver_success":[1,1,1],
            "reward":[-1.,-.3,-.2],
        })
        got=metrics_from_rows(log,.9,7.98)
        self.assertAlmostEqual(got["discounted_cost_from_start"],1+.9*.3+.81*.2)
        self.assertAlmostEqual(got["mean_cte_post_m"],.015)
        self.assertAlmostEqual(got["max_cte_post_m"],.02)
        self.assertEqual(got["solver_success_pct"],100.)

    def test_actual_one_step_casadi_frozen_evaluation_no_updates(self):
        import baseline as b
        from train_rl_environment import TrainEnvironment
        q,r,p=LEARNING_INITIAL.arrays()
        lo,hi=screening_box(.2)
        actor=GaussianWeightActor(
            np.random.default_rng(0),q,r,lo[0],hi[0],lo[1],hi[1],
            sigma_lower=.02,sigma_initial=.1,sigma_upper=.3)
        critic=SixParameterTDCritic(p,lo[2],hi[2])
        env=TrainEnvironment(
            train_length_m=float(b.CAD_S_END),
            q_initial=q,r_initial=r,p_initial=p,
            task_qe=Q_EVAL_CANDIDATE,task_re=R_EVAL_CANDIDATE,
            max_steps=1,ey=.05,heading_deg=0.)
        oldmu=actor.params["b_mu"].copy()
        oldp=critic.p.copy()
        summary,data=frozen_rollout(env,actor,critic,.9,
                                    "untrained_deterministic")
        self.assertEqual(len(data),1)
        self.assertEqual(summary["steps"],1)
        self.assertEqual(summary["solver_success_pct"],100.)
        self.assertTrue(summary["truncated"])
        self.assertEqual(summary["gamma"],.9)
        self.assertAlmostEqual(summary["discounted_cost_from_start"],
                               -float(data.reward.iloc[0]))
        np.testing.assert_array_equal(actor.params["b_mu"],oldmu)
        np.testing.assert_array_equal(critic.p,oldp)


if __name__=="__main__":
    unittest.main()
