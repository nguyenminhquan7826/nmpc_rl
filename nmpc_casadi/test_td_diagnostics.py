import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from td_diagnostics import episode_td_statistics
import experiment_plots as plots


class TDDiagnosticsTests(unittest.TestCase):
    def test_opposite_signs_cancel_mean_not_rms_or_weighted_signal(self):
        frame=pd.DataFrame({"td":[-2.,2.]})
        for i in range(6):frame[f"e_{i}"]=[2.,1.]
        result=episode_td_statistics(frame)
        self.assertEqual(result["mean_td"],0.)
        self.assertEqual(result["rms_td"],2.)
        self.assertEqual(result["mean_abs_td"],2.)
        self.assertEqual(result["negative_td_pct"],50.)
        self.assertEqual(result["positive_td_pct"],50.)
        self.assertEqual(result["mean_td_e2_4"],-3.)
        with self.assertRaises(ValueError):episode_td_statistics(frame.iloc[:0])

    def test_plot_signed_mean_separate_from_reward_and_rms(self):
        history=[]
        for ep,td in enumerate((-.8,-.2)):
            row={"episode":ep,"mean_reward":-1.,"mean_td":td,"rms_td":1.}
            for prefix,n in (("q_mean",6),("r_mean",2),("qf_final",6)):
                row.update({f"{prefix}_{i}":float(i+1) for i in range(n)})
            history.append(row)
        original=plots._save
        with tempfile.TemporaryDirectory() as tmp:
            def inspect(fig,target):
                if Path(target).name=="training_reward_td.png":
                    self.assertEqual(len(fig.axes),3)
                    np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(),[-.8,-.2])
                    np.testing.assert_allclose(fig.axes[1].lines[1].get_ydata(),[0,0])
                    np.testing.assert_allclose(fig.axes[2].lines[0].get_ydata(),[1,1])
                return original(fig,target)
            with patch.object(plots,"_save",side_effect=inspect):
                files=plots.plot_episode_weights(history,tmp)
            self.assertTrue(Path(files["learning"]).is_file())


if __name__=="__main__":unittest.main()
