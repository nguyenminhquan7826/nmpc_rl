import unittest
import tempfile
from pathlib import Path
import numpy as np
from nstep_actor import block_advantages,update_actor_block
from test_frozen_actor_critic_eval import fresh_checkpoint
from evaluate_controller_modes import load_checkpoint,initialize_adaptation,run_mode,evaluate_final_policy
from train_rl_environment import TrainEnvironment
import baseline as b

class NstepTests(unittest.TestCase):
    def test_returns_terminal_and_truncation(self):
        e=np.ones((3,6));p=np.ones(6);end=np.ones(6)*2
        adv,target=block_advantages([1,2,3],e,end,p,.5,True)
        np.testing.assert_allclose(target,[2.75,3.5,3.])
        np.testing.assert_allclose(adv,target+6)
        adv,target=block_advantages([1,2,3],e,end,p,.5,False)
        np.testing.assert_allclose(target,[-.25,-2.5,-9.])
        np.testing.assert_allclose(adv,target+6)

    def test_block_schedule_and_frozen_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.npz';fresh_checkpoint(path)
            def setup():
                a,c,cfg,lo,hi=load_checkpoint(path,4)
                a,c=initialize_adaptation(a,c,cfg,lo,hi,'initial_heads')
                cfg['actor_advantage_steps']=20
                env=TrainEnvironment(train_length_m=b.CAD_S_END,
                    q_initial=np.array(cfg['initial_Q']),r_initial=np.array(cfg['initial_R']),
                    p_initial=c.p.copy(),task_qe=np.array(cfg['Qe_fixed']),task_re=np.array(cfg['Re_fixed']),
                    max_steps=43,ey=.05,heading_deg=0.)
                return env,a,c,cfg
            env,a,c,cfg=setup();before={k:v.copy() for k,v in a.params.items()}
            summary,log=run_mode(env,a,c,cfg,'full_adapt')
            self.assertEqual(summary['actor_updates'],3)
            self.assertEqual(summary['actor_training_samples'],43)
            self.assertEqual(summary['critic_updates'],43)
            self.assertEqual(log.index[log.actor_update_applied].tolist(),[19,39,42])
            self.assertEqual(log.actor_batch_size[log.actor_update_applied].tolist(),[20,20,3])
            self.assertEqual(log.actor_advantage_horizon.tolist(),list(range(20,0,-1))*2+[3,2,1])
            self.assertTrue(np.isfinite(log.actor_advantage).all())
            self.assertTrue(any(not np.array_equal(before[k],a.params[k]) for k in before))
            env2,a2,c2,cfg2=setup()
            control,clog=run_mode(env2,a2,c2,cfg2,'critic_adapt_sampled')
            self.assertEqual(control['actor_updates'],0)
            # Same sampled policy and plant until the first block update.
            for key in ['q_4','r_1','X_m','Y_m','td','p_after_4']:
                np.testing.assert_array_equal(log[key].iloc[:20],clog[key].iloc[:20])
            for k in before:np.testing.assert_array_equal(before[k],a2.params[k])
            final={k:v.copy() for k,v in a.params.items()};pf=c.p.copy()
            score,_=evaluate_final_policy(env,a,c,cfg,'full_adapt')
            self.assertEqual(score['actor_updates'],0)
            for k in final:np.testing.assert_array_equal(a.params[k],final[k])
            np.testing.assert_array_equal(c.p,pf)

if __name__=='__main__':unittest.main()
