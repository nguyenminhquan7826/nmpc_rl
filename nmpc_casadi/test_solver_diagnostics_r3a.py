"""R3-A: verify solver diagnostic logging and unchanged R1 learning objective.

From nmpc_casadi:
    python -m unittest -v test_reward_r1 test_solver_diagnostics_r3a
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl_nmpc import (
    Agent, REWARD_SCHEME, export_solver_diagnostics,
)


class SolverDiagnosticsR3ATests(unittest.TestCase):
    def test_full_multiline_exception_jsonl_preserved(self):
        full_message = "QP failed: " + "detail " * 300 + "\nLast status: QP infeasible"
        failure = {
            "step_index": 4,
            "state_pre": [0.1, 0.0, 0.0, 0.0, 1.0, 2.0],
            "q_diag": [12.0, 3.0, 50.0, 6.0, 100.75, 100.75],
            "r_diag": [50.0, 10.0],
            "qf_diag": [240.0, 60.0, 1000.0, 120.0, 2015.0, 2015.0],
            "attempt_failures": [
                {"attempt": 1, "kind": "solver_exception",
                 "exception_type": "RuntimeError", "exception_full": full_message},
                {"attempt": 2, "kind": "rejected_solver_result",
                 "return_status": "Maximum_Iterations_Exceeded",
                 "constraint_violation": None},
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/"episode_0004_solver_failures.jsonl"
            export_solver_diagnostics(p, [failure])
            self.assertTrue(p.exists())
            rows = p.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
            recovered = json.loads(rows[0])
            self.assertEqual(recovered, failure)
            self.assertGreater(len(recovered["attempt_failures"][0]["exception_full"]), 300)

    def test_no_failure_produces_no_stale_file(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/"episode_0004_solver_failures.jsonl"
            export_solver_diagnostics(p, [])
            self.assertFalse(p.exists())
            p.write_text("stale", encoding="utf-8")
            export_solver_diagnostics(p, [])
            self.assertFalse(p.exists())

    def test_unmodified_r1_critic_mse_and_actor_lr_parameter(self):
        agent = Agent(seed=0, actor_lr=0.003, critic_lr=0.005, critic_nsteps=16)
        self.assertEqual(agent.actor_lr, 0.003)
        self.assertEqual(agent.critic_lr, 0.005)
        self.assertEqual(REWARD_SCHEME, "fixed_eval_progress_failure_v1")
        self.assertEqual(agent._meta()["version"], 10)
        e = np.array([0.01, 0., 0.01, 0., 0.02, -0.02])
        rows = []
        for i, rew in enumerate((-1., -1., -3000.)):
            s = np.zeros(12)
            s[0] = i * 0.1
            _, z = agent.propose(s, explore=True)
            rows.append((s, e, z, rew, s, e, float(i == 2)))
        result = agent.learn(rows)
        self.assertAlmostEqual(result["critic_loss"], result["critic_td_loss"])
        self.assertAlmostEqual(
            result["critic_loss_total"],
            result["critic_td_loss"] + result["critic_qf_reg_loss"],
        )
        self.assertNotIn("critic_td_huber_loss", result)
        self.assertTrue(all(np.isfinite(x).all() for x in agent.actor.p))
        self.assertTrue(all(np.isfinite(x).all() for x in agent.critic.p))


if __name__ == "__main__":
    unittest.main()
