"""Experiment D isolation: safety penalty and anti-saturation gradient.

Synthetic tests only; no private TRAIN or TEST data or trained checkpoints.
"""
import unittest
import numpy as np
from experiment_d_penalties import compute_training_reward
from gaussian_weight_actor import actor_mu_excess_squared
from actor_critic_updates import learn_one_transition
from test_component_learning_rates import new_agents


def diagnostic(success=True, terminal=False, reached=False):
    return {
        "solver_success":bool(success),
        "failure_terminal":bool(terminal),
        "reached_end":bool(reached),
    }


class DTrainingPenaltyTests(unittest.TestCase):
    def test_success_recovery_and_normal_reached_end_are_not_penalized(self):
        # Any eventually-successful solve (including fallback retry) is success.
        for reached in (False,True):
            v=compute_training_reward(-1.5,diagnostic(reached=reached),
                solver_failure_penalty=10, failure_terminal_penalty=100)
            self.assertEqual(v.paper_reward,-1.5)
            self.assertEqual(v.training_reward,-1.5)
            self.assertEqual(v.solver_penalty,0.)
            self.assertEqual(v.terminal_penalty,0.)

    def test_failure_step_and_catastrophic_terminal_are_distinct(self):
        x=compute_training_reward(-2,diagnostic(success=False),
            solver_failure_penalty=10, failure_terminal_penalty=100)
        y=compute_training_reward(-2,diagnostic(success=False,terminal=True),
            solver_failure_penalty=10, failure_terminal_penalty=100)
        self.assertEqual(x.training_reward,-12.)
        self.assertEqual(y.training_reward,-112.)
        self.assertEqual(y.terminal_penalty,100.)
        invalid=compute_training_reward(-2,diagnostic(success=True,terminal=True),
            solver_failure_penalty=10, failure_terminal_penalty=100)
        self.assertEqual(invalid.training_reward,-102.)

    def test_old_reward_exact_when_all_penalties_disabled(self):
        for successful in (True,False):
            v=compute_training_reward(-.38,diagnostic(successful),
                                      solver_failure_penalty=0,
                                      failure_terminal_penalty=0)
            self.assertEqual(v.training_reward,v.paper_reward)

    def test_invalid_settings_and_contradictory_status_are_rejected(self):
        for bad in (-1.,np.inf,np.nan):
            with self.assertRaises(ValueError):
                compute_training_reward(-.5,diagnostic(),
                                        solver_failure_penalty=bad)
        with self.assertRaises(ValueError):
            compute_training_reward(-1.,diagnostic(terminal=True,reached=True))
        with self.assertRaises(ValueError):
            compute_training_reward(-1.,{"solver_success":True})


class DActorMuRegularizationTests(unittest.TestCase):
    def test_regularization_exact_formula_on_positive_and_negative_mu(self):
        mu=np.array([3.,-4.,2.,-2.,0.,1.,-1.,2.1])
        self.assertAlmostEqual(float(actor_mu_excess_squared(mu,2.)),
                               1.+4.+.01)
        self.assertEqual(float(actor_mu_excess_squared(np.zeros(8),2.)),0.)

    def test_isolated_actor_regularization_backpropagates_to_shared_layers(self):
        a,_=new_agents()
        a.params["W_mu"][:3,:]=0.03
        a.params["b_mu"][:]=np.array([2.9,-3.5,2.5,-2.2,0.,0.,0.,0.])
        obs=np.linspace(-.6,.75,17)
        latent=a.propose(obs,deterministic=True).latent
        kwargs=dict(mu_regularization_coef=.02,mu_regularization_threshold=2.)
        old_loss,old_grad=a.loss_and_gradients(
            obs[None,:],latent[None,:],np.zeros(1))
        new_loss,g=a.loss_and_gradients(
            obs[None,:],latent[None,:],np.zeros(1),**kwargs)
        self.assertGreater(new_loss,old_loss)
        self.assertGreater(np.linalg.norm(g["W_mu"]),0.)
        self.assertGreater(np.linalg.norm(g["b_mu"]),0.)
        self.assertGreater(np.linalg.norm(g["W2"]),0.)
        self.assertGreater(np.linalg.norm(g["W1"]),0.)
        self.assertAlmostEqual(np.linalg.norm(g["W_sigma"]),0.,places=12)
        for param,idx in (("b_mu",(0,)),("W_mu",(0,0)),
                          ("W2",(0,0)),("W1",(0,0))):
            base=float(a.params[param][idx])
            eps=1e-5
            a.params[param][idx]=base+eps
            l_plus,_=a.loss_and_gradients(obs[None,:],latent[None,:],
                                          np.zeros(1),**kwargs)
            a.params[param][idx]=base-eps
            l_minus,_=a.loss_and_gradients(obs[None,:],latent[None,:],
                                           np.zeros(1),**kwargs)
            a.params[param][idx]=base
            self.assertAlmostEqual(g[param][idx],(l_plus-l_minus)/(2*eps),
                                   delta=1e-7)

    def test_regularization_not_rescaled_by_actor_head_lr_factors(self):
        a,_=new_agents()
        a.params["b_mu"][:]=3.
        obs=np.zeros((1,17))
        latent=np.zeros((1,8))
        reg=dict(mu_regularization_coef=.002,mu_regularization_threshold=2.)
        loss1,g1=a.loss_and_gradients(
            obs,latent,np.zeros(1),output_gradient_scale=np.ones(8),**reg)
        loss2,g2=a.loss_and_gradients(
            obs,latent,np.zeros(1),output_gradient_scale=np.full(8,3.),**reg)
        self.assertEqual(loss1,loss2)
        for name in g1:
            np.testing.assert_array_equal(g1[name],g2[name])

    def test_train_update_uses_shaped_reward_and_regularized_actor(self):
        a,c=new_agents()
        obs=np.linspace(-1.,1.,17)
        latent=a.propose(obs).latent
        e=np.array([.02,.04,.05,.02,.01,.08])
        paper_reward=-2.
        shaped=compute_training_reward(paper_reward,
            diagnostic(success=False,terminal=True),
            solver_failure_penalty=10., failure_terminal_penalty=100.)
        step=learn_one_transition(
            a,c,obs,latent,e,e,shaped.training_reward,
            gamma=.9,actor_lr=1e-5,critic_lr=1e-4,terminated=True,
            actor_mu_reg_coef=1e-3,actor_mu_reg_threshold=2.)
        self.assertAlmostEqual(step.reward,-112.)
        self.assertTrue(np.isfinite(step.td))
        self.assertTrue(np.isfinite(step.actor_loss))

    def test_invalid_mu_regularization_arguments(self):
        a,_=new_agents()
        obs=np.zeros((1,17)); z=np.zeros((1,8))
        for coefficient,threshold in ((-1.,2.),(.01,0.),
                                      (np.inf,2.),(.001,np.nan)):
            with self.assertRaises(ValueError):
                a.loss_and_gradients(
                    obs,z,np.array([-.2]),
                    mu_regularization_coef=coefficient,
                    mu_regularization_threshold=threshold)


if __name__=="__main__":
    unittest.main()
