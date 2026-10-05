"""Regression tests for the critic-only experiment. Run: python -m unittest test_critic_only -v."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rl_nmpc as rl


def transitions(agent):
    rng=np.random.default_rng(123)
    rows=[]
    for k in range(24):
        s=rng.normal(0,.3,12);sn=s+rng.normal(0,.01,12)
        e=rng.normal(0,.05,6);en=e+rng.normal(0,.005,6)
        _,z=agent.propose(s,True)
        rows.append((s,e,z,-.2-float(np.dot(e,e)),sn,en,float(k==23)))
    return rows


class CriticOnlyTests(unittest.TestCase):
    def test_actor_parameters_optimizer_and_noise_are_frozen(self):
        agent=rl.Agent(train_component='critic')
        # A warm-started/nonzero Adam must also remain untouched.
        agent.actor.t=7
        for x in agent.actor.m:x.fill(.2)
        for x in agent.actor.v:x.fill(.1)
        before=[x.copy() for x in agent.actor.p+agent.actor.m+agent.actor.v]
        rng_before=json.dumps(agent.rng.bit_generator.state)
        for _ in range(3):
            update=agent.learn(transitions(agent))
            self.assertEqual(update['actor_updated'],0)
            self.assertEqual(update['actor_parameter_delta'],0.)
            self.assertEqual(update['actor_loss'],0.)
            self.assertGreater(update['critic_parameter_delta'],0.)
        for actual,expected in zip(agent.actor.p+agent.actor.m+agent.actor.v,before):
            np.testing.assert_array_equal(actual,expected)
        self.assertEqual(agent.actor.t,7)
        self.assertEqual(agent.critic.t,3)
        self.assertEqual(json.dumps(agent.rng.bit_generator.state),rng_before)
        # Even corrupted Actor output cannot affect applied Q/R in this mode.
        agent.actor.p[-1].fill(100.)
        for s in [np.zeros(12),np.ones(12)]:
            q,r,f=agent.propose(s,True)[0]
            np.testing.assert_array_equal(q,rl.Q_INIT)
            np.testing.assert_array_equal(r,rl.R_INIT)
            self.assertTrue(np.all(np.isfinite(f)))
            self.assertTrue(np.all(f>=rl.FLO*rl.F_BASE))
            self.assertTrue(np.all(f<=rl.FHI*rl.F_BASE))

    def test_checkpoint_restores_critic_mode_and_independent_initial_rng(self):
        agent=rl.Agent(seed=4,train_component='critic')
        agent.initial_rng.uniform(size=10)
        agent.learn(transitions(agent))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.npz';agent.save(path)
            restored=rl.Agent(seed=9,train_component='critic',initial_seed=4)
            restored.load(path)
            self.assertEqual(restored.actor.t,0)
            self.assertEqual(restored.critic.t,1)
            np.testing.assert_array_equal(agent.initial_rng.uniform(size=10),restored.initial_rng.uniform(size=10))
            for lhs,rhs in zip(agent.critic.p+agent.critic.m+agent.critic.v,
                               restored.critic.p+restored.critic.m+restored.critic.v):
                np.testing.assert_array_equal(lhs,rhs)
            with self.assertRaisesRegex(ValueError,'configuration mismatch'):
                rl.Agent().load(path)
            with self.assertRaisesRegex(ValueError,'configuration mismatch'):
                rl.Agent(train_component='critic',critic_nsteps=1,initial_seed=4).load(path)
            legacy=Path(folder)/'legacy.npz';rl.Agent().save(legacy)
            rl.Agent().load(legacy)
            with self.assertRaisesRegex(ValueError,'configuration mismatch'):
                restored.load(legacy)

    def test_initial_conditions_are_independent_of_actor_exploration(self):
        critic=rl.Agent(seed=4,train_component='critic',initial_seed=42)
        both=rl.Agent(seed=4,train_component='both',initial_seed=42)
        for _ in range(5):
            for _ in range(17):both.propose(np.zeros(12),True)
            np.testing.assert_array_equal(critic.initial_rng.uniform(size=2),both.initial_rng.uniform(size=2))

    def test_controller_routing_and_target_boundary(self):
        agent=rl.Agent(train_component='critic')
        self.assertEqual(rl.learned_controller(agent),'critic_only')
        self.assertEqual(rl.evaluation_controllers(agent),['baseline','fixed_qf','critic_only'])
        target,h=rl.n_step_targets([-1,-2,-3],[-10,-20,-30],[0,1,0],3,gamma=.9)
        np.testing.assert_allclose(target,[-2.8,-2.,-30.])
        np.testing.assert_array_equal(h,[2,1,1])
        both=rl.Agent();both.learn(transitions(both))
        self.assertEqual(both.actor.t,1)
        self.assertEqual(both.critic.t,1)


if __name__=='__main__':unittest.main()
