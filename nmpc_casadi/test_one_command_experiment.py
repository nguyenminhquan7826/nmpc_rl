"""One-command layout and plots: no external manifest/audit paths.

Tests use synthetic arrays for plotting. No TEST data is read.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import experiment_config as cfg
import run_experiment as runner
from experiment_plots import (
    plot_episode_weights,plot_trajectories,plot_frozen_comparison)
from training_initialization import LEARNING_INITIAL, screening_box


class MinimalLayoutTests(unittest.TestCase):
    def test_single_run_command_requires_no_paths_or_manifest(self):
        args=runner.train_argv((1,10,25,50,100))
        self.assertIn("--use-internal-box",args)
        self.assertNotIn("--screen-manifest",args)
        self.assertNotIn("--reward-audit-summary",args)
        self.assertEqual(args[args.index("--gamma")+1],"0.9")
        self.assertEqual(args[args.index("--episodes")+1],"100")
        self.assertEqual(args[-5:],["1","10","25","50","100"])

    def test_layout_has_only_two_new_output_roots_and_pinned_source(self):
        self.assertEqual(cfg.TRAIN_CSV.parent,cfg.TRAJECTORY_DIR)
        self.assertEqual(cfg.RESULTS_DIR.name,"results")
        self.assertEqual(cfg.IMAGES_DIR.name,"images")
        self.assertEqual(cfg.GAMMA,.9)
        self.assertEqual(cfg.EPISODES,100)

    def test_fixed_actor_preserves_qr_and_cannot_sample(self):
        q,r,_=LEARNING_INITIAL.arrays()
        fixed=runner.FixedWeightActor(q,r)
        val=fixed.propose(np.zeros(17),deterministic=True)
        np.testing.assert_array_equal(val.q_diagonal,q)
        np.testing.assert_array_equal(val.r_diagonal,r)
        val.q_diagonal[0]=10000.
        self.assertAlmostEqual(fixed.q[0],q[0])
        with self.assertRaisesRegex(ValueError,"no stochastic"):
            fixed.propose(np.zeros(17),deterministic=False)

    def test_reject_overwrite_of_existing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp)/"results"
            r.mkdir()
            (r/"prior.csv").write_text("historic,do_not_delete",encoding="utf-8")
            i=Path(tmp)/"images"
            with patch.object(cfg,"RESULTS_DIR",r),patch.object(cfg,"IMAGES_DIR",i):
                with self.assertRaisesRegex(FileExistsError,"Preserve prior"):
                    runner._validate_config()
            self.assertEqual((r/"prior.csv").read_text(),"historic,do_not_delete")

    def test_plot_six_q_six_qf_and_two_r_into_images_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir=Path(tmp)/"images"
            history=[]
            for ep in range(5):
                row={
                    "episode":ep,
                    "mean_reward":-.5+.01*ep,
                    "rms_td":.5-.01*ep,
                }
                for i in range(6):
                    row[f"q_mean_{i}"]=10.+i+.01*ep
                    row[f"qf_final_{i}"]=1000.+i+ep
                for i in range(2):
                    row[f"r_mean_{i}"]=2.+i+.01*ep
                history.append(row)
            paths=plot_episode_weights(history,image_dir)
            self.assertEqual(len(paths),4)
            for path in paths.values():
                self.assertTrue(Path(path).is_file())
                self.assertGreater(Path(path).stat().st_size,1000)
            # Do not overwrite figures as this hides previous experiments.
            with self.assertRaises(FileExistsError):
                plot_episode_weights(history,image_dir)

    def test_real_xy_plot_uses_logged_state_and_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            ref=root/"train_xref.csv"
            pd.DataFrame({
                "X_ref_m":np.linspace(0,1,10),
                "Y_ref_m":np.linspace(0,.3,10),
            }).to_csv(ref,index=False)
            track=pd.DataFrame({
                "X_m":np.linspace(0,1,10),
                "Y_m":np.linspace(.05,.32,10),
            })
            image_dir=root/"images"
            image=plot_trajectories(
                ref,{"trained_episode_0100":track},image_dir)
            self.assertTrue(Path(image).is_file())
            self.assertGreater(Path(image).stat().st_size,1000)
            comparison=pd.DataFrame({
                "controller":["untrained_deterministic","strong_fixed_baseline",
                              "stage_lower_terminal_upper","trained_episode_0001",
                              "trained_episode_0100"],
                "mean_cte_post_m":[.02,.019,.017,.0199,.018],
            })
            image2=plot_frozen_comparison(comparison,image_dir)
            self.assertTrue(Path(image2).is_file())


if __name__=="__main__":
    unittest.main()
