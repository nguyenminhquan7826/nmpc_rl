"""Headless plotting regression tests; run from nmpc_casadi."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from plot_direct_p_logs import (
    plot_training_history, plot_training_episode,
    plot_training_output, plot_evaluation_output,
)


def sample_log():
    n = 8
    df = pd.DataFrame({
        "t_s": np.arange(n, dtype=float) * .071,
        "progress_m": np.arange(n, dtype=float) * .02,
        "X": np.arange(n, dtype=float) * .02,
        "Y": np.linspace(.05, .02, n),
        "X_ref": np.arange(n, dtype=float) * .02,
        "Y_ref": np.zeros(n),
        "cte_m": np.linspace(.05, .02, n),
        "e_psi": np.linspace(.01, -.01, n),
        "delta": np.linspace(.05, .02, n),
    })
    for i in range(6):
        df[f"q_{i}"] = 50.+i
        df[f"qf_{i}"] = 200.+10*i
    for i in range(2):
        df[f"r_{i}"] = 10.+i
    return df


class PlotTests(unittest.TestCase):
    def test_train_history_and_episode_pngs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = sample_log()
            updates = pd.DataFrame({
                "step": [4, 8],
                "td_rms": [1.5, 1.2],
                "td_mean": [-1.2, -1.],
                "critic_loss_total": [.4, .3],
                "actor_loss_total": [2., 1.],
            })
            hist = pd.DataFrame({
                "episode": [0, 1],
                "return": [-50., -40.],
                "mean_cte_post_m": [np.nan, .04],
                "rmse_heading_post_rad": [np.nan, .03],
                "solver_success_pct": [100., 100.],
            })
            for i in range(6):
                hist[f"p_final_{i}"] = [240.+i, 245.+i]
            hist.to_csv(root / "training_history.csv", index=False)
            log.to_csv(root / "episode_0000.csv", index=False)
            updates.to_csv(root / "episode_0000_updates.csv", index=False)
            plot_training_output(root)
            for filename in ("training_return.png",
                             "training_mean_cte_post.png",
                             "training_terminal_weights.png",
                             "plots/episode_0000_trajectory.png",
                             "plots/episode_0000_cte_response.png",
                             "plots/episode_0000_weights_qrqf.png",
                             "plots/episode_0000_td_losses.png"):
                p = root / filename
                self.assertTrue(p.is_file(), filename)
                self.assertGreater(p.stat().st_size, 1000, filename)

    def test_heldout_all_comparison_pngs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = sample_log()
            controllers = ("fixed_nmpc", "direct_p_frozen_sampled",
                           "direct_p_online_sampled")
            profiles = ("nominal", "payload")
            rows = []
            for profile in profiles:
                for controller in controllers:
                    log.to_csv(root / f"{profile}_{controller}.csv", index=False)
                    rows.append({
                        "profile": profile, "controller": controller,
                        "mean_cte_post_m": .04, "max_cte_post_m": .09,
                        "rmse_heading_post_rad": .02,
                        "solver_success_pct": 100.,
                        "core_loop_p95_ms": 2.,
                    })
                pd.DataFrame({"step": [1, 2], "td_rms": [2., 1.],
                              "td_mean": [-2., -1.],
                              "critic_loss_total": [2., 1.],
                              "actor_loss_total": [2., 1.]}).to_csv(
                    root / f"{profile}_direct_p_online_sampled_updates.csv",
                    index=False)
            pd.DataFrame(rows).to_csv(root / "comparison.csv", index=False)
            plot_evaluation_output(root)
            for filename in (
                "tracking_metric_comparison.png",
                "timing_fixed_vs_ac.png",
                "nominal_trajectory.png",
                "nominal_fixed_vs_ac_response.png",
                "nominal_fixed_vs_ac.png",
                "payload_trajectory.png",
                "payload_fixed_vs_ac_response.png",
                "nominal_direct_p_frozen_sampled_weights_qrqf.png",
                "nominal_direct_p_online_sampled_weights_qrqf.png",
                "nominal_online_td_losses.png",
            ):
                p = root / filename
                self.assertTrue(p.is_file(), filename)
                self.assertGreater(p.stat().st_size, 1000, filename)


if __name__ == "__main__":
    unittest.main()
