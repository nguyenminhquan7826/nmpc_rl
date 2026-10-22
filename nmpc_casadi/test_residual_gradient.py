import unittest
import numpy as np
from direct_p_critic import SixParameterTDCritic
from test_actor_critic_training import make_agent
from actor_critic_updates import learn_one_transition


class ResidualGradientTests(unittest.TestCase):
    def test_finite_difference_of_sampled_loss_and_projection(self):
        p=np.linspace(1.,2.,6);e=np.linspace(.1,.6,6);en=e[::-1].copy()
        reward=-.7;gamma=.99;lr=.01
        for terminated in (False,True):
            critic=SixParameterTDCritic(p,np.full(6,.1),np.full(6,4.))
            def loss(v):
                delta=reward+v@e**2-(0. if terminated else gamma*(v@en**2))
                return .5*delta**2
            numeric=[]
            for i in range(6):
                shift=np.zeros(6);shift[i]=1e-6
                numeric.append((loss(p+shift)-loss(p-shift))/2e-6)
            result=critic.td_step(e,en,reward,gamma,lr,terminated=terminated,
                                  update_mode="residual_gradient")
            np.testing.assert_allclose(result.gradient,numeric,rtol=1e-6,atol=1e-9)
            np.testing.assert_allclose(critic.p,p-lr*np.array(numeric),atol=1e-9)
            self.assertLess(loss(critic.p),loss(p))
        critic=SixParameterTDCritic(p,np.full(6,.1),np.full(6,2.1))
        critic.td_step(e,en,reward,gamma,1e6,update_mode="residual_gradient")
        self.assertTrue(np.all(critic.p>=.1) and np.all(critic.p<=2.1))

    def test_same_preupdate_td_and_actor_update_in_both_modes(self):
        results=[];actors=[];critics=[]
        for mode in ("semi_gradient","residual_gradient"):
            actor,critic=make_agent()
            obs=np.linspace(-.4,.4,17)
            latent=actor.propose(obs).latent
            result=learn_one_transition(actor,critic,obs,latent,
                np.linspace(.01,.06,6),np.linspace(.02,.07,6),-1.,
                gamma=.99,actor_lr=1e-5,critic_lr=1.,critic_update_mode=mode)
            results.append(result);actors.append(actor);critics.append(critic)
        self.assertEqual(results[0].td,results[1].td)
        for k in actors[0].params:
            np.testing.assert_array_equal(actors[0].params[k],actors[1].params[k])
        self.assertFalse(np.array_equal(critics[0].p,critics[1].p))
        with self.assertRaises(ValueError):
            critics[0].td_step(np.ones(6),np.ones(6),-1,.9,1,update_mode="unknown")


if __name__=="__main__":unittest.main()
