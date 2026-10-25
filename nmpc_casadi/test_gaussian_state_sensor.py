"""Measurement-noise reproducibility and true-vs-measured plant contract."""
import unittest
import numpy as np
from gaussian_state_sensor import GaussianStateSensor, EXAMPLE_STD


class GaussianStateSensorTests(unittest.TestCase):
    def test_same_seed_and_step_give_identical_noise_across_modes(self):
        a=GaussianStateSensor(EXAMPLE_STD,seed=123)
        b=GaussianStateSensor(EXAMPLE_STD,seed=123)
        state=np.array([.1,.01,.2,.05,3.,4.])
        for k in (0,1,4,200):
            ma,na=a.measure(state,k)
            mb,nb=b.measure(state,k)
            np.testing.assert_array_equal(ma,mb)
            np.testing.assert_array_equal(na,nb)
            np.testing.assert_allclose(ma-na,state,atol=1e-14)
        np.testing.assert_array_equal(a.noise_at(4),a.noise_at(4))
        self.assertFalse(np.array_equal(a.noise_at(0),a.noise_at(1)))
        self.assertFalse(np.array_equal(a.noise_at(1),GaussianStateSensor(EXAMPLE_STD,124).noise_at(1)))

    def test_zero_noise_leaves_plant_identical_and_rejects_bad_input(self):
        sensor=GaussianStateSensor([0]*6,seed=0)
        state=np.array([.1,.0,.0,.0,5.,-2.])
        measured,noise=sensor.measure(state,42)
        np.testing.assert_array_equal(measured,state)
        np.testing.assert_array_equal(noise,np.zeros(6))
        measured[0]=100
        self.assertEqual(state[0],.1)
        for bad in ([1]*5,[1]*7,[0,0,np.nan,0,0,0],[0,-1,0,0,0,0]):
            with self.assertRaises(ValueError):GaussianStateSensor(bad)
        with self.assertRaises(ValueError):sensor.noise_at(-1)
        with self.assertRaises(ValueError):sensor.measure([1]*5,0)

    def test_environment_uses_noisy_control_state_but_true_scoring(self):
        import baseline as b
        from train_rl_environment import TrainEnvironment
        from training_initialization import LEARNING_INITIAL
        from audit_training_reward import Q_EVAL_CANDIDATE, R_EVAL_CANDIDATE
        from direct_p_critic import SixParameterTDCritic
        q,r,p=LEARNING_INITIAL.arrays()
        sensor=GaussianStateSensor(EXAMPLE_STD,seed=7)
        env=TrainEnvironment(
            train_length_m=b.CAD_S_END,q_initial=q,r_initial=r,p_initial=p,
            task_qe=Q_EVAL_CANDIDATE,task_re=R_EVAL_CANDIDATE,
            max_steps=2,ey=.05,heading_deg=0.,sensor=sensor)
        obs=env.reset()
        np.testing.assert_allclose(obs.physical-obs.true_physical,
                                   sensor.noise_at(0),atol=1e-14)
        self.assertGreater(np.linalg.norm(obs.physical-obs.true_physical),0.)
        critic=SixParameterTDCritic(p,p*.5,p*2)
        before=env.x.copy()
        nxt,reward,terminated,truncated,diag=env.step(q,r,critic,obs)
        np.testing.assert_array_equal(obs.true_physical,before)
        self.assertAlmostEqual(reward,diag["reward_observed"])
        from rl_paper_core import paper_reward
        expected=paper_reward(obs.true_error,
                              np.array([diag["du_a"],diag["du_delta"]]),
                              Q_EVAL_CANDIDATE,R_EVAL_CANDIDATE)
        self.assertAlmostEqual(float(expected),diag["reward_true"])
        expected_next=np.asarray(b.Fd_plant(before,
            np.array([diag["a"],diag["steering"]]),b.p_nominal),float).reshape(-1)
        np.testing.assert_allclose(env.x,expected_next,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(nxt.physical-nxt.true_physical,sensor.noise_at(1),
                                   atol=1e-14)
        self.assertAlmostEqual(diag["cte_m"],obs.true_reference["projection"]["distance"])

if __name__=="__main__": unittest.main()
