import tempfile
import unittest
from pathlib import Path
import numpy as np
import baseline as b
from test_frozen_actor_critic_eval import fresh_checkpoint
from evaluate_controller_modes import load_checkpoint,run_mode,MODES,plot_comparison
from train_rl_environment import TrainEnvironment
from training_initialization import GOOD_NMPC
from direct_p_critic import SixParameterTDCritic
from run_experiment import FixedWeightActor


class ModeTests(unittest.TestCase):
    def test_all_modes_with_real_solver_and_parameter_freezes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/"model.npz";fresh_checkpoint(path)
            traces={}
            for mode in MODES+("frozen_sampled",):
                actor,critic,cfg,low,high=load_checkpoint(path,23)
                cfg.update(actor_lr=1e-5,critic_lr=1.)
                if mode=="fixed":
                    q,r,p=GOOD_NMPC.arrays()
                    actor=FixedWeightActor(q,r)
                    critic=SixParameterTDCritic(p,low[2],high[2])
                before={k:v.copy() for k,v in actor.params.items()};p0=critic.p.copy()
                env=TrainEnvironment(train_length_m=b.CAD_S_END,
                    q_initial=np.array(cfg["initial_Q"]),r_initial=np.array(cfg["initial_R"]),
                    p_initial=np.array(cfg["initial_P"]),
                    task_qe=np.array(cfg["Qe_fixed"]),task_re=np.array(cfg["Re_fixed"]),
                    max_steps=2,ey=.05,heading_deg=0.)
                summary,log=run_mode(env,actor,critic,cfg,mode)
                changed=any(not np.array_equal(before[k],v) for k,v in actor.params.items())
                self.assertEqual(changed,mode in ("actor_adapt","full_adapt"))
                self.assertEqual(not np.array_equal(p0,critic.p),mode in ("critic_adapt","full_adapt"))
                self.assertEqual(len(log),2)
                self.assertTrue((log.loop_ms>=log.solve_ms).all())
                traces[mode]=log
                if mode not in ("fixed","frozen"):
                    self.assertEqual(summary["actor_updates"],2 if mode in ("actor_adapt","full_adapt") else 0)
                    self.assertEqual(summary["critic_updates"],2 if mode in ("critic_adapt","full_adapt") else 0)
            plot_comparison(traces,root)
            self.assertTrue((root/"mode_comparison.png").is_file())


if __name__=="__main__":unittest.main()
