"""Independent reward audit and Actor/Critic one-transition contract."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from audit_training_reward import (
    Q_EVAL_CANDIDATE, R_EVAL_CANDIDATE, reward_diagnostics,
)
from actor_critic_updates import learn_one_transition
from direct_p_critic import SixParameterTDCritic
from gaussian_weight_actor import GaussianWeightActor
from rl_paper_core import paper_reward
from train_actor_critic import read_screened_box
from training_initialization import LEARNING_INITIAL, TRAIN_REFERENCE_SHA256, screening_box


def make_agent():
    q,r,p=LEARNING_INITIAL.arrays()
    low,high=screening_box(.2)
    actor=GaussianWeightActor(
        np.random.default_rng(432),q,r,low[0],high[0],low[1],high[1],
        sigma_lower=.02,sigma_initial=.1,sigma_upper=.3)
    critic=SixParameterTDCritic(p,low[2],high[2])
    return actor,critic


class TrainingContractTests(unittest.TestCase):
    def test_old_reward_weights_fixed_and_separate(self):
        np.testing.assert_allclose(Q_EVAL_CANDIDATE,[20,20,100,20,600,600])
        np.testing.assert_allclose(R_EVAL_CANDIDATE,[50,100])
        e=np.array([.01,0.,.02,0.,.01,-.01])
        du=np.array([.05,-.01])
        reward=paper_reward(e,du,Q_EVAL_CANDIDATE,R_EVAL_CANDIDATE)
        self.assertAlmostEqual(reward,-(
            .01**2*20+.02**2*100+.01**2*600+
            (-.01)**2*600+.05**2*50+(-.01)**2*100))

    def test_audit_from_real_log_field_names_and_td_without_last(self):
        data=pd.DataFrame({
            "e_vx":[.01,.02,.015],"e_vy":[.0,.0,.0],
            "e_psi":[.1,.03,-.04],"e_r":[0.,0.,0.],
            "e_X":[.02,.01,.01],"e_Y":[.01,.01,.03],
            "du_a":[.1,0.,-.02],"du_delta":[.03,.01,0.],
        })
        d=reward_diagnostics(data,gamma=.99)
        self.assertEqual(d["steps"],3)
        self.assertEqual(d["td_transitions_nonterminal"],2)
        self.assertTrue(d["mean_reward"]<0)
        self.assertGreater(d["td_rms"],0)
        self.assertGreater(d["mean_tracking_penalty"],0)
        self.assertGreater(d["mean_control_increment_penalty"],0)

    def test_audit_refuses_missing_state_error_fields(self):
        with self.assertRaisesRegex(ValueError,"Missing"):
            reward_diagnostics(pd.DataFrame({"du_a":[0.],"du_delta":[0.]}),gamma=.99)

    def test_shared_preupdate_td_no_change_for_zero_td(self):
        actor,critic=make_agent()
        observation=np.linspace(-.5,.5,17)
        latent=actor.propose(observation).latent
        e=np.zeros(6)
        pbefore=critic.p.copy()
        params_before={k:v.copy() for k,v in actor.params.items()}
        result=learn_one_transition(
            actor,critic,observation,latent,
            e,e,0.,gamma=.99,actor_lr=.001,critic_lr=.001)
        self.assertAlmostEqual(result.td,0)
        self.assertAlmostEqual(result.actor_loss,0)
        self.assertAlmostEqual(result.critic_loss,0)
        np.testing.assert_allclose(critic.p,pbefore)
        for name,old in params_before.items():
            np.testing.assert_allclose(actor.params[name],old)

    def test_nonterminal_same_td_updates_both_learned_heads_and_critic(self):
        actor,critic=make_agent()
        o=np.linspace(-1.,1.,17)
        sampled=actor.propose(o).latent+np.linspace(-.04,.07,8)
        e=np.array([.07,.01,.06,.02,.06,-.05])
        en=np.array([.04,.01,.02,.01,.04,-.04])
        p0=critic.p.copy()
        sig0=actor.params["b_sigma"].copy()
        mu0=actor.params["b_mu"].copy()
        value_now=critic.value(e)
        value_next=critic.value(en)
        expected=-1.2+.99*value_next-value_now
        step=learn_one_transition(
            actor,critic,o,sampled,e,en,-1.2,
            gamma=.99,actor_lr=.0001,critic_lr=.001,terminated=False)
        self.assertAlmostEqual(step.td,expected)
        self.assertAlmostEqual(step.value_current,value_now)
        self.assertAlmostEqual(step.value_next,value_next)
        self.assertGreater(step.actor_grad_norm,0)
        self.assertGreater(np.linalg.norm(actor.params["b_mu"]-mu0),0)
        self.assertGreater(np.linalg.norm(actor.params["b_sigma"]-sig0),0)
        self.assertGreater(np.linalg.norm(critic.p-p0),0)

    def test_terminal_drops_bootstrap_and_truncation_keeps_it(self):
        obs=np.zeros(17)
        error=np.array([.03,.04,.02,.01,.03,.02])
        nxt=np.array([.06,.04,.04,.02,.03,.01])
        actor1,critic1=make_agent()
        actor2,critic2=make_agent()
        latent=np.zeros(8)
        dterm=learn_one_transition(
            actor1,critic1,obs,latent,error,nxt,-.4,
            gamma=.99,actor_lr=.001,critic_lr=.001,terminated=True)
        dtrunc=learn_one_transition(
            actor2,critic2,obs,latent,error,nxt,-.4,
            gamma=.99,actor_lr=.001,critic_lr=.001,terminated=False)
        self.assertEqual(dterm.value_next,0.)
        self.assertNotEqual(dtrunc.value_next,0.)
        self.assertNotAlmostEqual(dterm.td,dtrunc.td)

    def test_no_implicit_learning_rates_or_non_boolean_termination(self):
        a,c=make_agent()
        with self.assertRaises(ValueError):
            learn_one_transition(
                a,c,np.zeros(17),np.zeros(8),np.zeros(6),np.zeros(6),-1.,
                gamma=.99,actor_lr=0.,critic_lr=.01)
        with self.assertRaises(ValueError):
            learn_one_transition(
                a,c,np.zeros(17),np.zeros(8),np.zeros(6),np.zeros(6),-1.,
                gamma=.99,actor_lr=.001,critic_lr=.01,terminated=1)

    def test_full_27_pass_screening_manifest_required(self):
        lower,upper=screening_box(.2)
        data={
            "train_sha256":TRAIN_REFERENCE_SHA256,
            "profile":"full","total_samples":27,"pass_samples":27,
            "empirical_all_pass":True,
            "screen_candidate_lower":[x.tolist() for x in lower],
            "screen_candidate_upper":[x.tolist() for x in upper],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"manifest.json"
            path.write_text(json.dumps(data))
            lo,hi,_=read_screened_box(path)
            self.assertEqual(lo[0].shape,(6,))
            self.assertEqual(hi[2].shape,(6,))
            data["train_sha256"]="changed"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"27/27"):
                read_screened_box(path)
            data["train_sha256"]=TRAIN_REFERENCE_SHA256
            data["empirical_all_pass"]=False
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"27/27"):
                read_screened_box(path)


if __name__=="__main__":
    unittest.main()
