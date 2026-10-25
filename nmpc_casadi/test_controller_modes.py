import tempfile
import unittest
from pathlib import Path
import numpy as np
import baseline as b
from test_frozen_actor_critic_eval import fresh_checkpoint
from evaluate_controller_modes import load_checkpoint,run_mode,MODES,plot_comparison,initial_fixed_policy,initialize_adaptation,evaluate_final_policy,plot_actor_diagnostics
from train_rl_environment import TrainEnvironment
from training_initialization import GOOD_NMPC
from direct_p_critic import SixParameterTDCritic
from run_experiment import FixedWeightActor


class ModeTests(unittest.TestCase):
    def test_custom_eval_configuration_is_synchronized_and_serializable(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"source.npz";fresh_checkpoint(path)
            original=path.read_bytes()
            actor,critic,cfg,low,high=load_checkpoint(path,0)
            self.assertEqual(len(cfg["initial_Q"]),6)
            self.assertEqual(cfg["initial_R"],[15,100])
            self.assertEqual(cfg["R_upper"],[30.,800.])
            self.assertGreater(cfg["critic_lr"],0.)
            np.testing.assert_array_equal(actor.lower,np.r_[low[0],low[1]])
            np.testing.assert_array_equal(actor.upper,np.r_[high[0],high[1]])
            np.testing.assert_array_equal(critic.lower,low[2])
            np.testing.assert_array_equal(critic.upper,high[2])
            np.testing.assert_allclose(cfg["fixed_Qf"],2*np.array(cfg["initial_Q"]))
            json.dumps(cfg)
            self.assertEqual(path.read_bytes(),original)
            with self.assertRaisesRegex(ValueError,"initial_heads"):
                initialize_adaptation(actor,critic,cfg,low,high,"trained")

    def test_fixed_terminal_is_independent_of_critic_initial_P(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"source.npz";fresh_checkpoint(path)
            actor,critic,cfg,low,high=load_checkpoint(path,12)
            self.assertIsNot(cfg["fixed_Qf"],cfg["initial_P"])
            fixed_qf=np.array(cfg["fixed_Qf"])
            # Changing only Critic initialization must not change Fixed.
            cfg["initial_P"]=(fixed_qf*.99).tolist()
            fixed_actor,fixed_terminal=initial_fixed_policy(cfg,low,high)
            _,adaptive_critic=initialize_adaptation(actor,critic,cfg,low,high,"initial_heads")
            np.testing.assert_array_equal(fixed_terminal.p,fixed_qf)
            np.testing.assert_allclose(adaptive_critic.p,fixed_qf*.99)
            # Live TD updates cannot mutate Fixed terminal weights/config.
            adaptive_critic.td_step(np.full(6,.05),np.full(6,.04),-1.,.9,.01)
            np.testing.assert_array_equal(fixed_terminal.p,fixed_qf)
            np.testing.assert_array_equal(cfg["fixed_Qf"],fixed_qf)

    def test_reset_heads_preserves_hidden_layers_and_matches_fixed_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"source.npz";fresh_checkpoint(path)
            source=path.read_bytes()
            for start in ("initial_heads","fresh"):
                actor,critic,cfg,low,high=load_checkpoint(path,12)
                hidden={k:actor.params[k].copy() for k in ("W1","b1","W2","b2")}
                actor,critic=initialize_adaptation(actor,critic,cfg,low,high,start)
                if start=="initial_heads":
                    for k,v in hidden.items():np.testing.assert_array_equal(v,actor.params[k])
                for obs in (np.zeros(17),np.linspace(-1,1,17)):
                    proposal=actor.propose(obs,deterministic=True)
                    np.testing.assert_allclose(proposal.q_diagonal,cfg["initial_Q"])
                    np.testing.assert_allclose(proposal.r_diagonal,cfg["initial_R"])
                    np.testing.assert_allclose(proposal.sigma,cfg["sigma_initial"])
                np.testing.assert_array_equal(critic.p,cfg["initial_P"])
            self.assertEqual(source,path.read_bytes())

    def test_all_modes_with_real_solver_and_parameter_freezes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/"model.npz";fresh_checkpoint(path)
            traces={}
            for mode in MODES+("frozen_sampled","critic_adapt_sampled"):
                actor,critic,cfg,low,high=load_checkpoint(path,23)
                cfg.update(actor_lr=1e-5,critic_lr=1.)
                if mode=="fixed":
                    actor,critic=initial_fixed_policy(cfg,low,high)
                    np.testing.assert_array_equal(actor.q,cfg["initial_Q"])
                    np.testing.assert_array_equal(actor.r,cfg["initial_R"])
                    np.testing.assert_array_equal(critic.p,cfg["initial_P"])
                elif mode in ("frozen","actor_adapt","critic_adapt","full_adapt","frozen_sampled","critic_adapt_sampled"):
                    actor,critic=initialize_adaptation(actor,critic,cfg,low,high,"initial_heads")
                before={k:v.copy() for k,v in actor.params.items()};p0=critic.p.copy()
                env=TrainEnvironment(train_length_m=b.CAD_S_END,
                    q_initial=np.array(cfg["initial_Q"]),r_initial=np.array(cfg["initial_R"]),
                    p_initial=np.array(cfg["initial_P"]),
                    task_qe=np.array(cfg["Qe_fixed"]),task_re=np.array(cfg["Re_fixed"]),
                    max_steps=2,ey=.05,heading_deg=0.)
                summary,log=run_mode(env,actor,critic,cfg,mode)
                changed=any(not np.array_equal(before[k],v) for k,v in actor.params.items())
                self.assertEqual(changed,mode in ("actor_adapt","full_adapt"))
                self.assertEqual(not np.array_equal(p0,critic.p),mode in ("critic_adapt","full_adapt","critic_adapt_sampled"))
                self.assertEqual(len(log),2)
                self.assertTrue((log.loop_ms>=log.solve_ms).all())
                traces[mode]=log
                if mode=="critic_adapt_sampled":
                    self.assertEqual(summary["score_role"],"online_adaptation")
                    self.assertEqual(summary["critic_updates"],2)
                    self.assertEqual(summary["actor_updates"],0)
                    np.testing.assert_allclose(log.q_det_4,cfg["initial_Q"][4])
                    self.assertFalse(np.allclose(log.q_4,log.q_det_4))
                    for i in range(8):self.assertIn(f"mu_{i}",log)
                    for i in range(17):self.assertIn(f"obs_{i}",log)
                if mode=="full_adapt":
                    # Deliberately move the final policy/P so resetting heads is detectable.
                    actor.params["b_mu"][0]+=.1
                    critic._p[0]+=.1
                    final_params={k:v.copy() for k,v in actor.params.items()}
                    final_p=critic.p.copy()
                    rng_before=repr(actor.rng.bit_generator.state)
                    final_summary,final_log=evaluate_final_policy(env,actor,critic,cfg,mode)
                    self.assertEqual(final_summary["actor_updates"],0)
                    self.assertEqual(final_summary["critic_updates"],0)
                    self.assertEqual(final_summary["score_role"],"post_adaptation_deterministic")
                    self.assertEqual(final_summary["mode"],mode)
                    self.assertNotAlmostEqual(final_log.q_0.iloc[0],cfg["initial_Q"][0])
                    np.testing.assert_allclose(final_log.p_0,final_p[0])
                    for k,v in final_params.items():np.testing.assert_array_equal(actor.params[k],v)
                    np.testing.assert_array_equal(critic.p,final_p)
                    self.assertEqual(repr(actor.rng.bit_generator.state),rng_before)
                    _,repeat=evaluate_final_policy(env,actor,critic,cfg,mode)
                    np.testing.assert_array_equal(final_log.q_0,repeat.q_0)
                    np.testing.assert_array_equal(final_log.reward,repeat.reward)
                if mode=="frozen":
                    for prefix,key in (("q","initial_Q"),("r","initial_R"),("p","initial_P")):
                        values=log[[f"{prefix}_{i}" for i in range(len(cfg[key]))]].to_numpy()
                        np.testing.assert_allclose(values,np.tile(cfg[key],(len(log),1)),atol=1e-10)
                    # Same physical weights and deterministic actions as Fixed.
                    np.testing.assert_allclose(log[["X_m","Y_m","reward"]],
                        traces["fixed"][["X_m","Y_m","reward"]],rtol=1e-6,atol=1e-8)
                if mode not in ("fixed","frozen"):
                    self.assertEqual(summary["actor_updates"],2 if mode in ("actor_adapt","full_adapt") else 0)
                    self.assertEqual(summary["critic_updates"],2 if mode in ("critic_adapt","full_adapt","critic_adapt_sampled") else 0)
            plot_comparison(traces,root)
            plot_actor_diagnostics(traces["full_adapt"],root)
            self.assertEqual({p.name for p in root.glob('*.png')},
                {"trajectories.png","cte_mean_max.png","full_adapt_weights.png","actor_learning_diagnostics.png"})


if __name__=="__main__":unittest.main()
