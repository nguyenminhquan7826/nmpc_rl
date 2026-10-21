"""One real CasADi NMPC solve with Gaussian Actor and direct-P TD update.

Uses the checked-in base reference solely as a CI ENGINEERING smoke test,
NOT as the pinned TRAIN trajectory and NOT as evidence of improved control.
"""
import unittest
import numpy as np

import baseline as b
from train_rl_environment import TrainEnvironment
from actor_critic_updates import learn_one_transition
from gaussian_weight_actor import GaussianWeightActor
from direct_p_critic import SixParameterTDCritic
from training_initialization import LEARNING_INITIAL, screening_box
from rl_paper_core import paper_reward


class JointNMPCSmokeTests(unittest.TestCase):
    def test_actor_cost_to_nmpc_plant_reward_and_td(self):
        q,r,p=LEARNING_INITIAL.arrays()
        low,high=screening_box(.20)
        actor=GaussianWeightActor(
            np.random.default_rng(19),q,r,low[0],high[0],
            low[1],high[1],
            sigma_lower=.02,sigma_initial=.1,sigma_upper=.3)
        critic=SixParameterTDCritic(p,low[2],high[2])
        qe=np.array([20.,20.,100.,20.,600.,600.])
        re=np.array([50.,100.])
        env=TrainEnvironment(
            train_length_m=float(b.CAD_S_END),
            q_initial=q,r_initial=r,p_initial=p,
            task_qe=qe,task_re=re,
            max_steps=1,ey=.05,heading_deg=0.)
        obs=env.observe()
        proposed=actor.propose(obs.features,deterministic=True)
        np.testing.assert_allclose(proposed.q_diagonal,q,atol=1e-12)
        np.testing.assert_allclose(proposed.r_diagonal,r,atol=1e-12)
        after,reward,terminal,truncated,details=env.step(
            proposed.q_diagonal,proposed.r_diagonal,critic,obs)
        self.assertTrue(details["solver_success"],
                        "Real adaptive NMPC solve should succeed")
        self.assertEqual(after.features.shape,(17,))
        self.assertTrue(np.all(np.isfinite(after.features)))
        self.assertTrue(np.isfinite(reward))
        expected=paper_reward(
            obs.error, [details["du_a"],details["du_delta"]],qe,re)
        self.assertAlmostEqual(reward,expected)
        self.assertTrue(truncated or terminal)
        initial_critic=critic.p.copy()
        step=learn_one_transition(
            actor,critic,obs.features,proposed.latent,obs.error,after.error,
            reward,gamma=.99,actor_lr=.000001,critic_lr=.001,
            terminated=terminal)
        self.assertTrue(np.isfinite(step.td))
        self.assertAlmostEqual(step.critic_loss,0.5*step.td**2)
        self.assertEqual(actor.parameter_count,6352)
        self.assertEqual(critic.p.shape,(6,))
        self.assertTrue(np.all(critic.p>0))
        self.assertTrue(np.all(critic.p>=low[2]))
        self.assertTrue(np.all(critic.p<=high[2]))
        self.assertTrue(np.all(np.isfinite(initial_critic)))


if __name__=="__main__":
    unittest.main()
