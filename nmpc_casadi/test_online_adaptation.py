"""Online adaptation updates real agents without touching the source model."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import baseline as b
from test_frozen_actor_critic_eval import fresh_checkpoint
from evaluate_frozen_actor_critic import load_frozen_agents
from train_rl_environment import TrainEnvironment
from online_adaptation import run_from_checkpoint
from training_initialization import LEARNING_INITIAL
from audit_training_reward import Q_EVAL_CANDIDATE, R_EVAL_CANDIDATE


class OnlineTests(unittest.TestCase):
    def test_real_transition_updates_both_and_preserves_source_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ckpt=root/"source.npz"
            low,high=fresh_checkpoint(ckpt)
            # Fixture lacks training LRs: add explicit online learning settings.
            with np.load(ckpt,allow_pickle=False) as z:
                arrays={k:z[k].copy() for k in z.files}
            meta=json.loads(str(arrays["metadata"].item()))
            meta.update(actor_lr=1e-5,critic_lr=1.,critic_update_mode="residual_gradient",
                        actor_output_lr_factors=[1.]*8,critic_p_lr_factors=[1.]*6)
            arrays["metadata"]=np.array(json.dumps(meta));np.savez_compressed(ckpt,**arrays)
            original=ckpt.read_bytes()
            q,r,p=LEARNING_INITIAL.arrays()
            env=TrainEnvironment(train_length_m=b.CAD_S_END,
                q_initial=q,r_initial=r,p_initial=p,
                task_qe=Q_EVAL_CANDIDATE,task_re=R_EVAL_CANDIDATE,
                max_steps=2,ey=.05,heading_deg=0.)
            out=root/"online"
            summary=run_from_checkpoint(env,ckpt,out,seed=17)
            self.assertEqual(ckpt.read_bytes(),original)
            log=pd.read_csv(out/"online_episode.csv")
            self.assertEqual(summary["parameter_updates"],len(log))
            self.assertTrue(log.actor_param_step_norm.sum()>0)
            self.assertTrue(log.critic_param_step_norm.sum()>0)
            np.testing.assert_allclose(log.filter(regex=r'^p_after_[0-5]$').iloc[0],
                                       log.filter(regex=r'^p_[0-5]$').iloc[1])
            with np.load(out/"checkpoint_after_online.npz",allow_pickle=False) as updated:
                self.assertTrue(any(not np.array_equal(arrays[k],updated[k])
                                    for k in arrays if k.startswith("actor_")))
                self.assertFalse(np.array_equal(arrays["critic_physical_P"],updated["critic_physical_P"]))
                np.testing.assert_allclose(updated["critic_physical_P"],
                    log.filter(regex=r'^p_after_[0-5]$').iloc[-1])
            for name in ("online_weights.png","online_trajectory.png","online_qf_change.png"):
                self.assertTrue((out/name).is_file())
            with self.assertRaises(FileExistsError):
                run_from_checkpoint(env,ckpt,out)


if __name__=="__main__":unittest.main()
