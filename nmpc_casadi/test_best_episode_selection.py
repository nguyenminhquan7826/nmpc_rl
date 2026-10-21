"""Best-episode selection and figures: deterministic, TRAIN-only, synthetic data."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from best_episode_selection import (
    select_best_train_episode, select_best_frozen_checkpoint,
)
from experiment_plots import plot_best_train_episode, plot_best_frozen_episode


def train(ep,cte,*,complete=True,solver=100.,reward=-2.):
    return {"episode":ep,"mean_cte_m":cte,"mean_reward":reward,
            "solver_success_pct":solver,"reached_end":complete}


def frozen(ep,cte,*,complete=True,solver=100.,failure=False,
           truncated=False,reward=-2.):
    return {"controller":f"trained_episode_{ep:04d}",
            "mean_cte_post_m":cte,"mean_reward":reward,
            "solver_success_pct":solver,"reached_end":complete,
            "failure_terminal":failure,"truncated":truncated}


class BestEpisodeTests(unittest.TestCase):
    def test_best_training_xy_uses_logged_positions_and_final_endpoint(self):
        import experiment_plots as plots
        df=pd.DataFrame({
            "progress_m":[0.,1.],"cte_m":[.05,.02],"reward":[-2.,-1.],
            "X_m":[0.,1.],"Y_m":[.05,.02],
            "X_ref_m":[0.,1.],"Y_ref_m":[0.,0.],
            "X_next_m":[1.,2.],"Y_next_m":[.02,.01],
        })
        original_save=plots._save
        with tempfile.TemporaryDirectory() as tmp:
            def check_and_save(fig,target):
                self.assertEqual(len(fig.axes),3)
                ref,actual=fig.axes[0].lines
                np.testing.assert_allclose(ref.get_ydata(),[0.,0.])
                np.testing.assert_allclose(actual.get_xdata(),[0.,1.,2.])
                np.testing.assert_allclose(actual.get_ydata(),[.05,.02,.01])
                return original_save(fig,target)
            with patch.object(plots,"_save",side_effect=check_and_save):
                self.assertTrue(Path(plot_best_train_episode(df,92,tmp)).is_file())
            df.loc[0,"X_m"]=np.nan
            with self.assertRaisesRegex(ValueError,"Non-finite"):
                plot_best_train_episode(df,92,tmp)

    def test_stochastic_best_requires_completion_and_solver_reliability(self):
        records=[
            train(0,.07),train(1,.001,complete=False),
            train(2,.002,solver=98.),train(3,.049),
            train(4,.050),train(5,.049,reward=-1.),
        ]
        best=select_best_train_episode(records)
        self.assertEqual(best["episode"],5)
        self.assertIsNone(select_best_train_episode([train(0,0.,complete=False)]))
        self.assertEqual(select_best_train_episode([train(0,.2)])["episode"],0)
        with self.assertRaises(ValueError):
            select_best_train_episode(records,101.)

    def test_frozen_best_only_qualified_checkpoints(self):
        records=[
            {"controller":"strong_fixed_baseline","mean_cte_post_m":.0001},
            frozen(1,.01,complete=False),
            frozen(5,.03,solver=98.),
            frozen(10,.05),
            frozen(25,.06),
            frozen(50,.001,failure=True,complete=False),
            frozen(100,.002,truncated=True,complete=False),
        ]
        self.assertEqual(select_best_frozen_checkpoint(records)["controller"],
                         "trained_episode_0010")
        self.assertIsNone(select_best_frozen_checkpoint([
            frozen(1,.1,complete=False)]))

    def test_best_train_plot_from_one_saved_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            n=10
            df=pd.DataFrame({
                "progress_m":np.linspace(0,7.9,n),
                "cte_m":np.linspace(.06,.02,n),
                "reward":np.linspace(-2.,-1.,n),
                "training_reward":np.linspace(-2.1,-1.1,n),
            })
            path=Path(plot_best_train_episode(df,14,tmp))
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size,1000)
            with self.assertRaises(FileExistsError):
                plot_best_train_episode(df,14,tmp)

    def test_best_frozen_real_xy_cte_and_qrp_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ref=root/"ref.csv"
            n=20
            pd.DataFrame({
                "X_ref_m":np.linspace(0,7.9,n),
                "Y_ref_m":.1*np.sin(np.linspace(0,4,n))
            }).to_csv(ref,index=False)
            p=np.linspace(0,7.9,n)
            log=pd.DataFrame({
                "X_m":p,
                "Y_m":.1*np.sin(p/2)+.02,
                "s_progress_m":p,
                "cross_track_m":np.full(n,.03)
            })
            for prefix,length,base in (("q",6,10.),("r",2,1.),("p",6,500.)):
                for i in range(length):
                    log[f"{prefix}_{i}"]=base+i+np.zeros(n)
            image_dir=root/"plots"
            files=plot_best_frozen_episode(ref,log,"trained_episode_0010",
                    image_dir,comparator_trace=log)
            self.assertEqual(set(files),{"xy","cte","weights"})
            for value in files.values():
                self.assertTrue(Path(value).is_file())
                self.assertGreater(Path(value).stat().st_size,1000)


if __name__=="__main__":
    unittest.main()
