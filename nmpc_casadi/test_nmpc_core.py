"""Unit checks for NMPC-only dynamics/reference and CLI outputs."""
import tempfile
import unittest
from pathlib import Path

import casadi as ca
import numpy as np
import pandas as pd

import baseline as b
from cad_eval_reference import prepare_cad_evaluation_reference
from nmpc_casadi import build_nmpc_solver, pack_parameters, tracking_error
from vehicle_model_casadi_variants import (
    build_continuous_model, build_rk4_discrete_model,
)


class NMPCOnlyTests(unittest.TestCase):
    def test_six_state_heading_wrap(self):
        x = ca.DM([.1, 0., -np.pi + .01, 0., 1., 2.])
        ref = ca.DM([.1, 0., np.pi - .01, 0., 1., 2.])
        e = np.asarray(tracking_error(x, ref)).reshape(-1)
        self.assertAlmostEqual(float(e[2]), .02, delta=1e-10)
        self.assertEqual(e.size, 6)

    def test_discrete_RK4_is_consistent(self):
        x = np.array([.1, 0, 0, 0, 0, 0], dtype=float)
        u = np.array([.2, 0], dtype=float)
        continuous = build_continuous_model("restoring")
        model = build_rk4_discrete_model(continuous, b.Ts, "unit_test_RK4")
        fx = np.asarray(model(x, u, b.p_nominal)).reshape(-1)
        self.assertEqual(fx.size, 6)
        self.assertTrue(np.isfinite(fx).all())
        self.assertGreater(fx[4], x[4])

    def test_solver_dimensions_and_fixed_costs(self):
        q, r, qf = b.get_robust_controller_weights()
        problem = build_nmpc_solver(b.Fd_controller, b.N, q, r, qf, b.cfg)
        self.assertEqual(problem["n_decision"], b.N * (6 + 2))
        xref = np.zeros((b.N + 1, 6))
        x0 = np.array([.1, 0, 0, 0, 0, 0], dtype=float)
        u_prev = np.zeros(2)
        p = pack_parameters(x0, u_prev, xref, b.p_nominal)
        self.assertEqual(len(p), problem["n_parameter"])

    def test_reference_preserves_input_speed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            s = np.linspace(0, 2.0, 81)
            vx = .08 + .02 * (s / s[-1])
            t = np.r_[0, np.cumsum(np.diff(s) / (.5 * (vx[:-1] + vx[1:])))]
            df = pd.DataFrame({
                "index": np.arange(len(s)), "s_m": s, "t_ref_s": t,
                "vx_ref_mps": vx, "vy_ref_mps": np.zeros_like(s),
                "psi_ref_rad": np.zeros_like(s), "r_ref_radps": np.zeros_like(s),
                "X_ref_m": s, "Y_ref_m": np.zeros_like(s),
                "kappa_1_per_m": np.zeros_like(s)
            })
            src = root / "xref.csv"
            df.to_csv(src, index=False)
            info, meta = prepare_cad_evaluation_reference(
                src, root / "out", .2)
            self.assertAlmostEqual(info["cad_s_end"], 2.)
            tail = pd.read_csv(meta["tail_file"])
            np.testing.assert_allclose(
                tail["vx_ref_mps"].to_numpy()[:len(df)],
                df["vx_ref_mps"].to_numpy(), atol=1e-15, rtol=1e-14)


if __name__ == "__main__":
    unittest.main()
