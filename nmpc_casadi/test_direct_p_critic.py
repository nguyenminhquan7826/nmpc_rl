"""Run from nmpc_casadi: python -m unittest -v test_direct_p_critic"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import rl_nmpc as rl
from direct_p_critic import DiagonalTDCritic, DirectPAgent
from ac_ablation import learn_selected_component


class DirectPTests(unittest.TestCase):
    def test_one_step_td_sign_and_projection(self):
        c=DiagonalTDCritic(np.ones(6),scale=np.ones(6),
                           min_multiplier=.5,max_multiplier=2.)
        e=np.array([[1.,0,0,0,0,0]])
        en=np.zeros_like(e)
        delta,target,value,grad=c.td_gradient(e,[-2.],en,[0.],.99)
        self.assertAlmostEqual(delta[0],-1.)
        self.assertAlmostEqual(grad[0],-1.)
        c.update([grad],.1)
        self.assertAlmostEqual(c.physical_diagonal()[0],1.1)
        c.update([np.array([-1000.,0.,0.,0.,0.,0.])],1.)
        self.assertAlmostEqual(c.physical_diagonal()[0],2.)
        self.assertTrue(np.all(c.physical_diagonal()>0))

    def test_terminated_drops_bootstrap_but_truncated_does_not(self):
        c=DiagonalTDCritic(np.ones(6),scale=np.ones(6),
                           min_multiplier=.5,max_multiplier=2.)
        e=np.zeros((2,6)); en=np.ones((2,6))
        delta,_,_,_=c.td_gradient(e,[-1.,-1.],en,[1.,0.],.9)
        np.testing.assert_allclose(delta,[-1.,-6.4])

    def test_only_six_parameters_and_physical_mapping(self):
        initial=rl.F_BASE.copy()
        agent=DirectPAgent()
        self.assertEqual(len(agent.critic.p),1)
        self.assertEqual(agent.critic.p[0].shape,(6,))
        np.testing.assert_allclose(agent.propose(np.zeros(12),False)[0][2],
                                   initial)
        e=np.array([.02,.01,.03,.02,.04,.05])
        expected=-np.dot(e*e,initial)
        self.assertAlmostEqual(float(agent.value(np.zeros(12),e)),expected)

    def test_critic_only_freezes_actor_weights_and_optimizer(self):
        agent=DirectPAgent()
        before=[a.copy() for a in agent.actor.p]
        rows=[]
        for j in range(8):
            s=np.ones(12)*.01
            e=np.ones(6)*.015
            sn=s+.001
            en=e*.95
            _,z=agent.propose(s,True)
            rows.append((s,e,z,-.8,sn,en,0.))
        old=agent.critic.p[0].copy()
        result=learn_selected_component(agent,rows,'critic')
        self.assertGreater(result['critic_parameter_delta'],0.)
        self.assertEqual(result['actor_parameter_delta'],0.)
        for a,b in zip(agent.actor.p,before):
            np.testing.assert_array_equal(a,b)
        self.assertFalse(np.array_equal(agent.critic.p[0],old))

    def test_checkpoint_roundtrip(self):
        agent=DirectPAgent(seed=4,critic_lr=.002)
        agent.critic.update([np.ones(6)],.002)
        with tempfile.TemporaryDirectory() as t:
            checkpoint=Path(t)/'direct.npz'
            agent.save(checkpoint)
            second=DirectPAgent(seed=9)
            second.load(checkpoint)
            np.testing.assert_array_equal(agent.critic.p[0],
                                          second.critic.p[0])
            self.assertEqual(agent.critic.t,second.critic.t)
            with self.assertRaisesRegex(ValueError,'configuration mismatch'):
                rl.Agent().load(checkpoint)


if __name__=='__main__':
    unittest.main()
