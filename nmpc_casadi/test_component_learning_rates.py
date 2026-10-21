"""Per-component Q/R/Qf learning-rate experiments must be safe and verifiable.

No change to state error, actor network architecture, fixed Eq4 reward,
Critic six-dimensional direct P, or NMPC dynamics/constraints.
"""
import unittest
import numpy as np

import experiment_config as cfg
from actor_critic_updates import learn_one_transition
from direct_p_critic import SixParameterTDCritic
from gaussian_weight_actor import GaussianWeightActor
from training_initialization import LEARNING_INITIAL,screening_box


def new_agents(seed=223):
    q,r,p=LEARNING_INITIAL.arrays()
    lo,hi=screening_box(.2)
    a=GaussianWeightActor(
        np.random.default_rng(seed),q,r,lo[0],hi[0],lo[1],hi[1],
        sigma_lower=.02,sigma_initial=.1,sigma_upper=.3)
    c=SixParameterTDCritic(p,lo[2],hi[2])
    return a,c


class IndependentLearningRateTests(unittest.TestCase):
    def test_user_config_d_reproduces_completed_v3_settings(self):
        self.assertEqual(tuple(cfg.ACTOR_OUTPUT_LR_FACTORS),
                         (1.,1.,1.,1.,2.5,2.5,1.,1.))
        self.assertEqual(tuple(cfg.CRITIC_P_LR_FACTORS),
                         (1.23,.86,.25,.41,2.5,2.5))
        self.assertEqual(cfg.CRITIC_LR,1e-4)
        self.assertEqual(cfg.ACTOR_LR,1e-4)
        self.assertEqual(cfg.GAMMA,.90)
        self.assertEqual(cfg.WEIGHT_BOX_MARGIN,.40)
        self.assertEqual(cfg.EPISODES,100)
        self.assertEqual(cfg.SOLVER_FAILURE_PENALTY,10.)
        self.assertEqual(cfg.FAILURE_TERMINAL_PENALTY,100.)
        self.assertEqual(cfg.ACTOR_MU_REG_COEF,1e-3)

    def test_actor_preconditioner_modifies_mu_sigma_and_shared_layers(self):
        a,_=new_agents()
        # Synthetic nonzero head weights make the shared-layer chain rule nontrivial.
        a.params["W_mu"][:3,:]=.021
        a.params["W_sigma"][:3,:]=-.013
        obs=np.linspace(-.5,.8,17)[None,:]
        z=np.linspace(-.4,.3,8)[None,:]
        td=np.array([-.7])
        loss,base=a.loss_and_gradients(obs,z,td)
        factors=np.array([2.,1.,1.,1.,1.,1.,.5,1.])
        new_loss,new=a.loss_and_gradients(
            obs,z,td,output_gradient_scale=factors)
        self.assertAlmostEqual(loss,new_loss)
        for name in ("b_mu","b_sigma"):
            np.testing.assert_allclose(new[name][0],base[name][0]*2.)
            np.testing.assert_allclose(new[name][6],base[name][6]*.5)
            np.testing.assert_allclose(new[name][1:6],base[name][1:6])
            np.testing.assert_allclose(new[name][7],base[name][7])
        for name in ("W_mu","W_sigma"):
            np.testing.assert_allclose(new[name][:,0],base[name][:,0]*2.)
            np.testing.assert_allclose(new[name][:,6],base[name][:,6]*.5)
            np.testing.assert_allclose(new[name][:,1:6],base[name][:,1:6])
            np.testing.assert_allclose(new[name][:,7],base[name][:,7])
        self.assertGreater(np.linalg.norm(new["W2"]-base["W2"]),0.)
        self.assertGreater(np.linalg.norm(new["W1"]-base["W1"]),0.)

    def test_actor_identity_equal_to_original_gradient(self):
        a,_=new_agents()
        obs=np.linspace(-1.,1.,17)[None,:]
        z=np.linspace(.1,.4,8)[None,:]
        oldloss,old=a.loss_and_gradients(obs,z,np.array([1.]))
        newloss,new=a.loss_and_gradients(
            obs,z,np.array([1.]),output_gradient_scale=np.ones(8))
        self.assertEqual(oldloss,newloss)
        for name in old:
            np.testing.assert_array_equal(old[name],new[name])

    def test_critic_scalar_vs_six_identity_rates(self):
        _,a=new_agents()
        _,b=new_agents()
        e=np.array([.05,.03,.04,.02,-.05,.01])
        en=.8*e
        x=a.td_step(e,en,-.8,.9,.01)
        y=b.td_step(e,en,-.8,.9,np.ones(6)*.01)
        self.assertEqual(x.td_error,y.td_error)
        np.testing.assert_array_equal(a.p,b.p)

    def test_critic_component_rates_scale_only_respective_axes(self):
        _,a=new_agents()
        _,b=new_agents()
        e=np.array([.1,.2,.3,.4,.05,.1])
        zero=np.zeros(6)
        p_before=a.p
        factors=np.array([3.,1.,.5,1.,4.,2.])
        step=a.td_step(e,zero,-3.,.9,np.asarray(factors)*.01)
        step2=b.td_step(e,zero,-3.,.9,.01)
        base_diff=b.p-p_before
        np.testing.assert_allclose(a.p-p_before,base_diff*factors,
                                   rtol=1e-8,atol=1e-12)
        self.assertAlmostEqual(step.td_error,step2.td_error)
        self.assertTrue(np.all(a.p>0))

    def test_actor_critic_shared_preupdate_td_with_nonuniform_rates(self):
        a,c=new_agents()
        obs=np.linspace(-1,1,17)
        latent=a.propose(obs).latent
        e=np.array([.03,.05,.1,.04,.08,.09])
        en=e*.9
        factors=np.array([2.,.5,1.,1.,1.,.8,1.5,2.])
        p_factors=np.array([3.,2.,.5,1.,4.,4.])
        before=c.p.copy()
        pcopy=SixParameterTDCritic(before,c.lower,c.upper)
        step=learn_one_transition(
            a,c,obs,latent,e,en,-1.,
            gamma=.9,actor_lr=1e-5,critic_lr=.001,
            actor_output_scales=factors,critic_component_scales=p_factors)
        pcopy.td_step(e,en,-1.,.9,.001*p_factors)
        self.assertTrue(np.isfinite(step.actor_loss))
        self.assertTrue(np.isfinite(step.critic_loss))
        self.assertGreater(step.actor_param_step_norm,0)
        np.testing.assert_array_equal(c.p,pcopy.p)

    def test_invalid_lr_shapes_and_nonfinite_rejected_without_update(self):
        a,c=new_agents()
        e=np.ones(6)*.1
        obs=np.zeros(17)
        latent=a.propose(obs).latent
        for invalid in ([1.]*5,[1.]*7+[0.],[1.]*7+[float("nan")]):
            with self.assertRaises(ValueError):
                a.loss_and_gradients(
                    obs[None,:],latent[None,:],np.array([-.1]),
                    output_gradient_scale=invalid)
        for invalid in ([1.]*5,[1.]*5+[-1.],[1.]*5+[np.inf]):
            with self.assertRaises(ValueError):
                c.td_step(e,e,-1.,.9,invalid)
        before=c.p.copy()
        params={k:v.copy() for k,v in a.params.items()}
        with self.assertRaises(ValueError):
            learn_one_transition(
                a,c,obs,latent,e,e,-1.,gamma=.9,
                actor_lr=1e-5,critic_lr=.001,
                actor_output_scales=[1.]*7,
                critic_component_scales=[1.]*6)
        np.testing.assert_array_equal(c.p,before)
        for k,v in params.items():
            np.testing.assert_array_equal(a.params[k],v)


if __name__=="__main__":
    unittest.main()
