"""Tests for 100-episode pinned-TRAIN checkpoint learning curve orchestration."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_training_curve import (
    validate_history, evaluation_command, summarise_history,
)
from training_initialization import TRAIN_REFERENCE_SHA256


def fixture(tmp, n=100, *, gamma=.9, signed=True):
    root=Path(tmp)
    root.mkdir(parents=True,exist_ok=True)
    doc={
        "reference_role":"TRAIN_ONLY",
        "train_sha256":TRAIN_REFERENCE_SHA256,
        "gamma":gamma,
        "reward_audit_verified":signed,
    }
    (root/"run_config.json").write_text(json.dumps(doc))
    frame=pd.DataFrame({
        "episode":np.arange(n),
        "steps":np.full(n,1048),
        "mean_reward":np.linspace(-.8,-.6,n),
        "rms_td":np.linspace(1.2,.8,n),
        "solver_success_pct":np.full(n,100.),
        "reached_end":np.full(n,True),
        "terminated":np.full(n,True),
        "truncated":np.full(n,False),
    })
    frame.to_csv(root/"training_history.csv",index=False)
    for ep in (1,10,25,50,100):
        if ep<=n:
            (root/f"checkpoint_ep{ep-1:04d}.npz").write_bytes(b"unit test mock")
    return frame


class MultiCheckpointEvaluationTests(unittest.TestCase):
    def test_valid_100_episode_selection_is_one_based_and_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            table=fixture(tmp)
            root,config,actual=validate_history(
                tmp,[1,10,25,50,100])
            self.assertEqual(len(actual),100)
            self.assertEqual(config["gamma"],.9)
            self.assertEqual(root,Path(tmp).resolve())
            summary=summarise_history(actual,[1,10,25,50,100])
            self.assertGreater(summary["train_mean_reward_delta_last_minus_first"],0)
            self.assertEqual(summary["train_episode_reached_end_count"],100)

    def test_missing_checkpoint_or_partial_history_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp,n=50)
            with self.assertRaisesRegex(ValueError,"not available"):
                validate_history(tmp,[100])
            with self.assertRaisesRegex(ValueError,"Missing checkpoint"):
                validate_history(tmp,[2])

    def test_wrong_gamma_or_unsigned_reward_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp,n=10,gamma=.99)
            with self.assertRaisesRegex(ValueError,"gamma"):
                validate_history(tmp,[1,10])
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp,n=10,signed=False)
            with self.assertRaisesRegex(ValueError,"gamma"):
                validate_history(tmp,[1,10])

    def test_nonconsecutive_episodes_and_nan_history_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame=fixture(tmp,n=10)
            frame.loc[4,"episode"]=17
            frame.to_csv(Path(tmp)/"training_history.csv",index=False)
            with self.assertRaisesRegex(ValueError,"Nonconsecutive"):
                validate_history(tmp,[1,10])
        with tempfile.TemporaryDirectory() as tmp:
            frame=fixture(tmp,n=10)
            frame.loc[4,"rms_td"]=np.nan
            frame.to_csv(Path(tmp)/"training_history.csv",index=False)
            with self.assertRaisesRegex(ValueError,"Nonfinite"):
                validate_history(tmp,[1,10])

    def test_frozen_replay_call_never_updates_or_switches_to_test(self):
        cmd=evaluation_command(
            reference="reference_split/train_xref.csv",
            screen_root="results_screen/full_20pct",
            train_root="results_rl/one",
            output="results_rl/checkpoints/epoch_0100",
            epoch=100,max_steps=1150,all_train_conditions=True)
        self.assertTrue(any("evaluate_frozen_actor_critic.py" in x for x in cmd))
        self.assertIn("checkpoint_ep0099.npz"," ".join(cmd))
        self.assertIn("--all-train-conditions",cmd)
        self.assertIn("--max-steps",cmd)
        self.assertNotIn("--train",cmd)
        self.assertNotIn("--test",cmd)
        self.assertEqual(cmd[cmd.index("--max-steps")+1],"1150")


if __name__=="__main__":
    unittest.main()
