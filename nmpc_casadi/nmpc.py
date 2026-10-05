import argparse
import gc
import itertools
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vehicle_model_casadi_variants import (
    build_continuous_model,
    build_rk4_discrete_model,
)
from nmpc_casadi import (
    build_nmpc_solver,
    pack_parameters,
    unpack_solution,
    shift_warm_start,
)
from reference_manager import ReferenceManager


# ============================================================
# 1) Fixed experiment configuration
# ============================================================
# EDIT these diagonal entries to test one fixed controller before RL.
# Q and Qf: vx, vy, psi, r, X, Y; R: Delta a, Delta delta.
Q_TEST = np.array([12., 3., 50., 6., 70.75, 70.75])
R_TEST = np.array([0.6, 70.])
QF_TEST = np.array([12., 3., 50., 6., 70.75, 70.75])

Ts = 0.071              # [s] 14.08 Hz
N = 10                   # fixed for the embedded-target study
TMAX = 120.0              # [s]
DEADLINE_MS = Ts * 1000.0
TEST_SPEED_MPS = 0.10      # [m/s] fixed reference speed for this experiment
ABORT_AFTER_CONSECUTIVE_SOLVER_FAILS = 3  # abort after 3 FINAL failures
ENABLE_FRESH_ROLLOUT_FALLBACK = True

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "sweep_results"
RESULTS.mkdir(parents=True, exist_ok=True)
CASES_DIR = RESULTS / "cases"
CASES_DIR.mkdir(parents=True, exist_ok=True)

ROBUST_CASES_DIR = RESULTS / "robustness_cases"
ROBUST_CASES_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_ROBUST_CASES_DIR = RESULTS / "initial_robustness_cases"
INITIAL_ROBUST_CASES_DIR.mkdir(parents=True, exist_ok=True)

BOUNDARY_CASES_DIR = RESULTS / "boundary_cases"
BOUNDARY_CASES_DIR.mkdir(parents=True, exist_ok=True)

SOLVER_DIAG_DIR = RESULTS / "solver_diagnostic"
SOLVER_DIAG_DIR.mkdir(parents=True, exist_ok=True)

COMBINED_CASES_DIR = RESULTS / "combined_robustness_cases"
COMBINED_CASES_DIR.mkdir(parents=True, exist_ok=True)

QRQF_REGION_CASES_DIR = RESULTS / "qrqf_region_cases"
QRQF_REGION_CASES_DIR.mkdir(parents=True, exist_ok=True)

QRQF_STRESS_CASES_DIR = RESULTS / "qrqf_stress_cases"
QRQF_STRESS_CASES_DIR.mkdir(parents=True, exist_ok=True)

# Exact Q/R/Qf tuning base used when case_071 was selected.
# IMPORTANT: this is intentionally NOT the later frozen controller Q/R.
QRQF_BASE_Q_DIAG = np.array([12.0, 3.0, 15.0, 6.0, 183.75, 192.5])
QRQF_BASE_R_DIAG = np.array([0.6, 16])
QRQF_SELECTED_CASE = "case_071"

# ============================================================
# Extended one-factor Q/R/Qf stress study
# ============================================================
# Purpose: deliberately widen the parameter range so the influence of each
# weighting group is visually obvious. These are sensitivity/stress points,
# NOT automatically recommended controller settings.
#
# Every study changes ONE factor at a time around case_071.
QRQF_STRESS_MULTIPLIERS = [1.0, 4.0, 8.0, 16.0, 25.0]
QRQF_STRESS_SELECTED = {
    "alpha_position": 1.0,
    "alpha_heading": 1.0,
    "beta_delta": 1.0,
    "gamma_terminal": 1.0,
}

# Prefer the numerically smoothed Xref if available.
REF_SOURCE = BASE / "xref_states_smoothed.csv"
if not REF_SOURCE.exists():
    REF_SOURCE = BASE / "xref_states_controller.csv"

# The reference tail is used only by the prediction horizon. The real CAD path
# still ends at CAD_S_END and the closed-loop simulation stops there.
TAIL_LENGTH_M = 0.20
TAIL_FILE = RESULTS / "_xref_with_virtual_tail.csv"

# Initial condition used for every Q/R/Qf candidate.
START_X = 0
START_Y = 0.05
START_VX = None
START_PSI = None
START_VY = 0.0
START_R = 0.0
START_S_HINT = None

# ============================================================
# 2) Vehicle Model V1 -- nominal parameters from the selected 1:10 platform
# ============================================================

# Parameter order: [m, Iz, Cf, Cr, lf, lr, mu_roll, g, eps_v]
# Wheelbase is deliberately fixed to 0.2515 m for this study.
WHEELBASE_M = 0.2515
LF_M = WHEELBASE_M * 132.0 / 252.0
LR_M = WHEELBASE_M * 120.0 / 252.0

p_nominal = np.array([
    23.55 / 9.81,  # m [kg] ~= 2.4006
    0.044,         # Iz [kg m^2]
    0.40,          # Cf [N/rad]
    0.35,          # Cr [N/rad]
    LF_M,          # lf [m]
    LR_M,          # lr [m]
    0.02,          # rolling-resistance coefficient
    9.81,          # g [m/s^2]
    0.01,          # low-speed regularization [m/s]
], dtype=float)

# For this nominal simulation, plant and controller use the same model.
# Later this can be deliberately perturbed for robustness testing.
p_plant = p_nominal.copy()

# Additional hardware metadata kept here for traceability.
VEHICLE_META = {
    "wheelbase_m": WHEELBASE_M,
    "wheel_radius_m": 0.0325,
    "track_width_m": 0.1605,
    "cg_height_m": 0.07213,
    "steering_hw_limit_rad": np.deg2rad(33.44),
    "hardware_speed_cap_mps": 0.60,
    "test_reference_speed_mps": TEST_SPEED_MPS,
}

# ============================================================
# 3) Hard constraints -- fixed during the whole sweep
# ============================================================
cfg = {
    # u = [a, delta]
    # Acceleration bounds remain provisional until actuator identification.
    "u_min": np.array([-2.0, -0.55]),
    "u_max": np.array([ 2.0,  0.55]),

    # Per-sample input increments at Ts = 0.02 s.
    "du_min": np.array([-0.20, -0.040]),
    "du_max": np.array([ 0.20,  0.040]),

    # State/model-domain bounds.
    "vx_min": 0.05,
    "vx_max": 0.15,
    "r_max": 4.0,
    "beta_max": 0.60,
    "vy_abs_max": 0.60,

    "sqp_max_iter": 7,
}

# ============================================================
# 4) Q/R/Qf sweep definition
# ============================================================
# Center point of the search. Qf = gamma_f * Q.
Q_CENTER = np.array([12.0, 3.0, 15.0, 6.0, 183.75, 192.5])
R_CENTER = np.array([0.6, 16])

ALPHA_POSITION = [1.25, 1.50, 1.75]
ALPHA_HEADING = [0.75, 1.00, 1.25]
BETA_STEERING_SMOOTHNESS = [1.0, 1.5, 2.0]
GAMMA_TERMINAL = [10.0, 15.0, 20.0]

# ============================================================
# Frozen controller for robustness study (selected case_071)
# ============================================================
# Controller model remains p_nominal in every robustness case.
# Only the simulated plant parameters are perturbed.
ROBUST_Q_DIAG = np.array([12.0, 3.0, 15.0, 6.0, 183.75, 192.5])
ROBUST_R_DIAG = np.array([0.6, 16.0])
ROBUST_GAMMA_TERMINAL = 15.0

# One-factor-at-a-time parameter uncertainty levels.
# Parameter order in p_nominal:
# [m, Iz, Cf, Cr, lf, lr, mu_roll, g, eps_v]
ROBUST_PARAM_SPECS = {
    "m":       {"index": 0, "pct": 0.10},
    "Iz":      {"index": 1, "pct": 0.10},
    "Cf":      {"index": 2, "pct": 0.20},
    "Cr":      {"index": 3, "pct": 0.20},
    "mu_roll": {"index": 6, "pct": 0.25},
}

# ============================================================
# Initial-condition robustness grid
# ============================================================
INITIAL_EY_M = [-0.10, -0.05, 0.0, 0.05, 0.10]
INITIAL_EPSI_DEG = [-10.0, -5.0, 0.0, 5.0, 10.0]

# Capture is declared only if BOTH position and heading enter the corridor
# and remain there continuously for the dwell time.
CAPTURE_CTE_M = 0.05
CAPTURE_HEADING_RAD = np.deg2rad(5.0)
CAPTURE_DWELL_S = 1.0

# ============================================================
# Local boundary diagnostic around the failed (+0.10 m, -10 deg) point
# ============================================================
BOUNDARY_EY_M = [0.06, 0.08, 0.10, 0.12, 0.14]
BOUNDARY_EPSI_DEG = [-14.0, -12.0, -10.0, -8.0, -6.0]
BOUNDARY_DIAGNOSTIC_EY_M = 0.10
BOUNDARY_DIAGNOSTIC_EPSI_DEG = -10.0
BOUNDARY_DIAGNOSTIC_SQP_ITERS = [7, 15]

# Solver/warm-start diagnostic at the isolated failed point.
SOLVER_DIAG_EY_M = 0.10
SOLVER_DIAG_EPSI_DEG = -10.0
SOLVER_DIAG_SQP_MAX_ITER = 7
SOLVER_DIAG_IPOPT_MAX_ITER = 200
SOLVER_DIAG_HEADING_GAIN = 1.0

# ============================================================
# Combined robustness study
# ============================================================
# Four initial-condition levels x four plant/model-mismatch scenarios = 16 cases.
#
# The controller prediction model ALWAYS remains p_nominal.
# Only the simulated plant is changed.
COMBINED_INITIAL_SCENARIOS = [
    {
        "name": "baseline",
        "ey_m": 0.05,
        "epsi_deg": 0.0,
    },
    {
        "name": "moderate",
        "ey_m": 0.05,
        "epsi_deg": -5.0,
    },
    {
        "name": "strong",
        "ey_m": 0.10,
        "epsi_deg": -10.0,
    },
    {
        "name": "strong_mirror",
        "ey_m": -0.10,
        "epsi_deg": 10.0,
    },
]

# The adverse directions are based on the preceding OFAT study:
#   m -10%      -> worse CTE/heading than m +10%
#   Cf +20%     -> worse mean CTE than Cf -20%
#   Cr +20%     -> worse heading RMSE than Cr -20%
#   mu +25%     -> largest mean-CTE degradation
#
# These are engineering stress envelopes, not experimentally identified
# confidence intervals.
COMBINED_PLANT_SCENARIOS = [
    {
        "name": "plant_nominal",
        "m_scale": 1.00,
        "Cf_scale": 1.00,
        "Cr_scale": 1.00,
        "mu_scale": 1.00,
    },
    {
        "name": "high_resistance",
        "m_scale": 1.00,
        "Cf_scale": 1.00,
        "Cr_scale": 1.00,
        "mu_scale": 1.25,
    },
    {
        "name": "lateral_stiff",
        "m_scale": 1.00,
        "Cf_scale": 1.20,
        "Cr_scale": 1.20,
        "mu_scale": 1.00,
    },
    {
        "name": "all_adverse",
        "m_scale": 0.90,
        "Cf_scale": 1.20,
        "Cr_scale": 1.20,
        "mu_scale": 1.25,
    },
]

# Feasibility criteria. The initial 0.15 m capture transient is excluded from
# the steady tracking criteria by evaluating only after this progress value.
EVAL_AFTER_S_M = 1.3
MIN_COMPLETION_PCT = 99.0
MIN_SOLVER_SUCCESS_PCT = 99.0
MAX_MEAN_CTE_POST_M = 0.050
MAX_CTE_POST_M = 0.100
MAX_RMSE_HEADING_POST_RAD = 0.20
MAX_HOST_DEADLINE_MISS_PCT = 1.0  # reported separately; not part of tracking feasibility

# Numerical tolerance used only when checking hard bounds in logged floating-point data.
FEAS_TOL = 1e-6


# ============================================================
# Safe plot saving (Windows-friendly)
# ============================================================
def _safe_filename(name: str) -> str:
    """Return a Windows-safe filename component."""
    name = str(name)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
    name = name.rstrip(" .")
    return name or "plot.png"


def save_figure_safe(fig, filename: str, dpi: int = 170):
    """Save a Matplotlib figure robustly on Windows.

    The function:
      1) sanitizes the filename,
      2) converts Path -> absolute string before passing it to Pillow/Matplotlib,
      3) avoids a collision if a directory happens to have the target .png name,
      4) falls back to a short folder in the user's home directory if Windows
         rejects the project output path.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_filename(filename)
    target = (RESULTS / safe_name).resolve()

    if target.exists() and target.is_dir():
        target = (RESULTS / f"{Path(safe_name).stem}_plot{Path(safe_name).suffix}").resolve()

    try:
        fig.savefig(str(target), dpi=dpi, bbox_inches="tight")
        print("Saved plot:", repr(str(target)))
        return target
    except OSError as exc:
        print("WARNING: primary plot save failed:", repr(str(target)))
        print("Reason:", repr(exc))

        fallback_dir = Path.home() / "nmpc_plots"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback = (fallback_dir / safe_name).resolve()

        if fallback.exists() and fallback.is_dir():
            fallback = (
                fallback_dir
                / f"{Path(safe_name).stem}_plot{Path(safe_name).suffix}"
            ).resolve()

        fig.savefig(str(fallback), dpi=dpi, bbox_inches="tight")
        print("Saved plot to fallback:", repr(str(fallback)))
        return fallback


# ============================================================
# Reference preparation / endpoint fix
# ============================================================
def prepare_reference_with_tail(
    source_csv: Path,
    output_csv: Path,
    tail_length_m: float
):
    """
    Prepare the reference trajectory for the experiment.

    - Force the real CAD reference to run at constant vx_ref = 0.1 m/s.
    - Set vy_ref = 0.
    - Recompute r_ref = vx_ref * kappa.
    - Recompute t_ref.
    - Append a virtual tail after the real CAD endpoint.
    """

    # TEST_SPEED_MPS is defined globally in the fixed experiment configuration.

    # ============================================================
    # 1. Load reference
    # ============================================================
    df = pd.read_csv(source_csv).copy()

    required = [
        "s_m",
        "vx_ref_mps",
        "vy_ref_mps",
        "psi_ref_rad",
        "r_ref_radps",
        "X_ref_m",
        "Y_ref_m",
        "kappa_1_per_m",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing Xref columns: {missing}")

    df = df.sort_values("s_m").reset_index(drop=True)

    # ============================================================
    # 2. Force the whole CAD reference to vx_ref = 0.1 m/s
    # ============================================================
    df["vx_ref_mps"] = TEST_SPEED_MPS

    df["vy_ref_mps"] = 0.0

    df["r_ref_radps"] = (
        TEST_SPEED_MPS
        * df["kappa_1_per_m"].to_numpy(dtype=float)
    )

    # ============================================================
    # 3. Recompute reference time t(s)
    #
    # dt = ds / v_ref
    # ============================================================
    s_arr = df["s_m"].to_numpy(dtype=float)

    t_ref = np.zeros(len(df), dtype=float)

    if len(df) > 1:
        ds_arr = np.diff(s_arr)

        t_ref[1:] = np.cumsum(
            ds_arr / TEST_SPEED_MPS
        )

    df["t_ref_s"] = t_ref

    # ============================================================
    # 4. Save information about the REAL CAD endpoint
    # ============================================================
    cad_s_end = float(df["s_m"].iloc[-1])

    cad_x = df["X_ref_m"].to_numpy(dtype=float)
    cad_y = df["Y_ref_m"].to_numpy(dtype=float)

    # Typical waypoint spacing
    ds_values = np.diff(s_arr)

    positive_ds = ds_values[ds_values > 1e-9]

    if len(positive_ds) == 0:
        raise ValueError("Invalid s_m: path length is zero.")

    ds = float(np.median(positive_ds))

    n_tail = max(
        2,
        int(np.ceil(tail_length_m / ds))
    )

    # ============================================================
    # 5. Initial condition for virtual tail
    # ============================================================
    last = df.iloc[-1].copy()

    s_now = float(last["s_m"])
    x_now = float(last["X_ref_m"])
    y_now = float(last["Y_ref_m"])
    psi_now = float(last["psi_ref_rad"])

    kappa0 = float(last["kappa_1_per_m"])

    # Because the complete real reference has already been forced
    # to 0.1 m/s, the tail also keeps 0.1 m/s.
    vx_end = TEST_SPEED_MPS
    vy_end = 0.0

    t_now = float(last["t_ref_s"])

    rows = []

    # ============================================================
    # 6. Generate virtual prediction tail
    # ============================================================
    for j in range(1, n_tail + 1):

        # Smoothly reduce curvature to zero
        frac = j / n_tail

        kappa_j = (1.0 - frac) * kappa0

        # Midpoint integration of path geometry
        psi_mid = psi_now + 0.5 * kappa_j * ds

        x_now += ds * np.cos(psi_mid)
        y_now += ds * np.sin(psi_mid)

        psi_now += kappa_j * ds
        s_now += ds

        row = last.copy()

        row["s_m"] = s_now

        row["vx_ref_mps"] = vx_end
        row["vy_ref_mps"] = vy_end

        row["psi_ref_rad"] = psi_now

        row["kappa_1_per_m"] = kappa_j

        # r_ref = v_ref * kappa
        row["r_ref_radps"] = vx_end * kappa_j

        row["X_ref_m"] = x_now
        row["Y_ref_m"] = y_now

        if "index" in df.columns:
            row["index"] = int(df["index"].iloc[-1]) + j

        # dt = ds / v
        t_now += ds / vx_end
        row["t_ref_s"] = t_now

        rows.append(row)

    # ============================================================
    # 7. Combine CAD path + virtual tail
    # ============================================================
    tail_df = pd.DataFrame(
        rows,
        columns=df.columns
    )

    out = pd.concat(
        [df, tail_df],
        ignore_index=True
    )

    out.to_csv(
        output_csv,
        index=False
    )

    # ============================================================
    # 8. Return information
    # ============================================================
    return {
        "cad_s_end": cad_s_end,
        "cad_X": cad_x,
        "cad_Y": cad_y,
        "tail_s_end": float(out["s_m"].iloc[-1]),
        "n_original": len(df),
        "n_tail": len(tail_df),
    }

REF_INFO = prepare_reference_with_tail(REF_SOURCE, TAIL_FILE, TAIL_LENGTH_M)
CAD_S_END = REF_INFO["cad_s_end"]


# ============================================================
# Fixed model / RK4 plant
# ============================================================
f_controller = build_continuous_model("restoring")
Fd_controller = build_rk4_discrete_model(f_controller, Ts, name="Fd_controller")
f_plant = build_continuous_model("restoring")
Fd_plant = build_rk4_discrete_model(f_plant, Ts, name="Fd_plant")


def make_reference_manager():
    return ReferenceManager(
        str(TAIL_FILE),
        enforce_monotonic=True,
        search_back_m=0.15,
        search_forward_m=1.2,
        stop_at_end=False,
        use_midpoint_progress=True,
    )


def build_initial_state_and_guess(
    rm,
    problem,
    initial_ey_m=None,
    initial_epsi_rad=None,
):
    """
    Build the initial state and a dynamics-consistent SQP guess.

    Default behavior keeps START_X/START_Y. For initial-condition robustness,
    initial_ey_m and initial_epsi_rad are offsets relative to the reference
    start pose.
    """
    if START_S_HINT is not None:
        rm.reset(START_S_HINT)

    if initial_ey_m is None and initial_epsi_rad is None:
        proj0 = rm.project_xy_to_path(START_X, START_Y)
        s0 = proj0["s"]
        ref0 = rm.interpolate_reference(s0)

        x = ref0.copy()
        x[4] = START_X
        x[5] = START_Y
        if START_PSI is not None:
            x[2] = START_PSI
    else:
        s0 = float(rm.s_start)
        ref0 = rm.interpolate_reference(s0)
        ey = 0.0 if initial_ey_m is None else float(initial_ey_m)
        epsi = 0.0 if initial_epsi_rad is None else float(initial_epsi_rad)

        # Reference normal n = [-sin(psi_ref), cos(psi_ref)]
        x = ref0.copy()
        x[4] = ref0[4] - ey * np.sin(ref0[2])
        x[5] = ref0[5] + ey * np.cos(ref0[2])
        x[2] = np.arctan2(
            np.sin(ref0[2] + epsi),
            np.cos(ref0[2] + epsi),
        )
        proj0 = rm.project_xy_to_path(x[4], x[5])

    x[1] = START_VY
    x[3] = START_R
    if START_VX is not None:
        x[0] = START_VX

    u_prev = np.array([p_nominal[6] * p_nominal[7], 0.0], dtype=float)

    initial_ref = rm.get_nmpc_reference(x[4], x[5], Ts, N)
    xref0 = initial_ref["xref"]
    kappa_h0 = rm.interpolate_curvature(initial_ref["s_horizon"][:-1])
    L = p_nominal[4] + p_nominal[5]
    delta_ff0 = np.arctan(L * kappa_h0)
    delta_ff0 = np.clip(delta_ff0, cfg["u_min"][1], cfg["u_max"][1])

    a_ff0 = np.full(N, p_nominal[6] * p_nominal[7])
    U_guess0 = np.vstack([a_ff0, delta_ff0])
    X_guess0 = np.zeros((6, N))

    x_guess = x.copy()

    for i in range(N):
        u_guess = U_guess0[:, i]

        x_guess = np.array(
            Fd_controller(
                x_guess,
                u_guess,
                p_nominal
            )
        ).astype(float).reshape(-1)

        X_guess0[:, i] = x_guess
    z_guess = np.concatenate([
        X_guess0.reshape(-1, order="F"),
        U_guess0.reshape(-1, order="F"),
    ])
    return x, u_prev, z_guess, proj0



def build_fresh_dynamics_rollout_guess(rm, x, u_prev, ref_result):
    """
    Reinitialize the SQP guess from the CURRENT state.

    This is used only after the shifted warm start fails.  The input guess uses
    curvature feedforward + rolling-resistance compensation and is projected
    sequentially onto the input/rate bounds.  The state guess is then generated
    by rolling the controller model forward, so it is dynamics-consistent.
    """
    x = np.asarray(x, dtype=float).reshape(6)
    u_prev = np.asarray(u_prev, dtype=float).reshape(2)

    s_horizon = ref_result["s_horizon"]
    kappa_h = rm.interpolate_curvature(s_horizon[:-1])

    L = p_nominal[4] + p_nominal[5]
    delta_ff = np.arctan(L * kappa_h)
    delta_ff = np.clip(
        delta_ff,
        cfg["u_min"][1],
        cfg["u_max"][1],
    )

    a_ff = np.full(
        N,
        p_nominal[6] * p_nominal[7],
        dtype=float,
    )

    U_guess = np.vstack([a_ff, delta_ff])

    # Sequentially enforce both absolute input bounds and Δu bounds.
    prev = u_prev.copy()
    for i in range(N):
        lower = np.maximum(
            cfg["u_min"],
            prev + cfg["du_min"],
        )
        upper = np.minimum(
            cfg["u_max"],
            prev + cfg["du_max"],
        )
        U_guess[:, i] = np.clip(U_guess[:, i], lower, upper)
        prev = U_guess[:, i].copy()

    X_guess = np.zeros((6, N), dtype=float)
    xg = x.copy()

    for i in range(N):
        xg = np.array(
            Fd_controller(
                xg,
                U_guess[:, i],
                p_nominal,
            )
        ).astype(float).reshape(-1)

        X_guess[:, i] = xg

    return np.concatenate([
        X_guess.reshape(-1, order="F"),
        U_guess.reshape(-1, order="F"),
    ])


def compute_metrics(log: pd.DataFrame, reached_end: bool, solver_fail_count: int):
    metrics = {}
    for name in ["vx", "vy", "psi", "r", "X", "Y"]:
        e = log[f"e_{name}"].to_numpy(float)
        metrics[f"rmse_{name}"] = float(np.sqrt(np.mean(e ** 2)))

    metrics["max_cross_track_m"] = float(log["cross_track_m"].max())
    metrics["mean_cross_track_m"] = float(log["cross_track_m"].mean())
    metrics["final_progress_m"] = float(log["s_progress_m"].iloc[-1])
    metrics["path_length_m"] = float(CAD_S_END)
    metrics["completion_pct"] = float(100.0 * log["s_progress_m"].iloc[-1] / CAD_S_END)
    metrics["sim_time_s"] = float(log["t_s"].iloc[-1])
    metrics["reached_end"] = bool(reached_end)
    # solver_success is the FINAL result after the optional fresh-rollout retry.
    metrics["solver_success_pct"] = float(100.0 * log["solver_success"].mean())

    if "primary_solver_success" in log.columns:
        metrics["primary_solver_success_pct"] = float(
            100.0 * log["primary_solver_success"].mean()
        )
    else:
        metrics["primary_solver_success_pct"] = metrics["solver_success_pct"]

    if "fallback_attempted" in log.columns:
        fallback_attempt_count = int(log["fallback_attempted"].sum())
        fallback_success_count = int(log["fallback_success"].sum())
        metrics["fallback_attempt_count"] = fallback_attempt_count
        metrics["fallback_success_count"] = fallback_success_count
        metrics["fallback_recovery_pct"] = (
            100.0 * fallback_success_count / fallback_attempt_count
            if fallback_attempt_count > 0 else np.nan
        )
    else:
        metrics["fallback_attempt_count"] = 0
        metrics["fallback_success_count"] = 0
        metrics["fallback_recovery_pct"] = np.nan

    if "primary_solve_ms" in log.columns:
        metrics["primary_solve_mean_ms"] = float(log["primary_solve_ms"].mean())
        metrics["primary_solve_max_ms"] = float(log["primary_solve_ms"].max())
    else:
        metrics["primary_solve_mean_ms"] = np.nan
        metrics["primary_solve_max_ms"] = np.nan

    if "fallback_attempted" in log.columns and bool(log["fallback_attempted"].any()):
        attempted = log[log["fallback_attempted"] == 1]
        metrics["fallback_solve_mean_ms"] = float(attempted["fallback_solve_ms"].mean())
        metrics["fallback_solve_max_ms"] = float(attempted["fallback_solve_ms"].max())
    else:
        metrics["fallback_solve_mean_ms"] = np.nan
        metrics["fallback_solve_max_ms"] = np.nan

    metrics["solve_mean_ms"] = float(log["solve_ms"].mean())
    metrics["solve_p95_ms"] = float(np.percentile(log["solve_ms"], 95))
    metrics["solve_p99_ms"] = float(np.percentile(log["solve_ms"], 99))
    metrics["solve_max_ms"] = float(log["solve_ms"].max())
    metrics["deadline_miss_pct_20ms"] = float(100.0 * np.mean(log["solve_ms"] > DEADLINE_MS))
    metrics["max_abs_delta_rad"] = float(np.max(np.abs(log["delta_cmd"])))
    metrics["max_abs_a_mps2"] = float(np.max(np.abs(log["a_cmd"])))
    metrics["max_abs_ddelta_rad"] = float(np.max(np.abs(log["du_delta"])))
    metrics["max_abs_da_mps2"] = float(np.max(np.abs(log["du_a"])))
    metrics["min_vx_mps"] = float(log["vx"].min())
    metrics["max_vx_mps"] = float(log["vx"].max())
    metrics["solver_fail_count"] = int(solver_fail_count)
    metrics["solver_aborted"] = bool(
        ("solver_fail_streak" in log.columns)
        and (log["solver_fail_streak"].max() >= ABORT_AFTER_CONSECUTIVE_SOLVER_FAILS)
    )
    if "solver_fail_streak" in log.columns:
        metrics["max_solver_fail_streak"] = int(log["solver_fail_streak"].max())
    else:
        metrics["max_solver_fail_streak"] = 0

    post = log[log["s_progress_m"] >= EVAL_AFTER_S_M]
    if len(post) == 0:
        post = log
    metrics["mean_cte_post_m"] = float(post["cross_track_m"].mean())
    metrics["max_cte_post_m"] = float(post["cross_track_m"].max())
    metrics["rmse_heading_post_rad"] = float(np.sqrt(np.mean(post["e_psi"].to_numpy(float) ** 2)))
    metrics["rmse_vx_post_mps"] = float(np.sqrt(np.mean(post["e_vx"].to_numpy(float) ** 2)))

    # ------------------------------------------------------------
    # Numerical-validity check
    # ------------------------------------------------------------
    # A failed SQP sample deliberately logs objective = NaN.  That NaN must NOT
    # make the whole trajectory "nonfinite" if the physical state/control log is
    # otherwise valid.  We therefore check only the variables that must always
    # remain finite, and require a finite objective only on successful solves.
    finite_columns = [
        "t_s", "s_progress_m", "cross_track_m", "solve_ms",
        "vx", "vy", "psi", "r", "X", "Y",
        "vx_ref", "vy_ref", "psi_ref", "r_ref", "X_ref", "Y_ref",
        "e_vx", "e_vy", "e_psi", "e_r", "e_X", "e_Y",
        "a_cmd", "delta_cmd", "du_a", "du_delta",
    ]
    finite_state_control_ok = bool(
        np.isfinite(log[finite_columns].to_numpy(dtype=float)).all()
    )

    successful_rows = log["solver_success"].to_numpy(dtype=bool)
    if np.any(successful_rows):
        objective_ok = bool(
            np.isfinite(
                log.loc[successful_rows, "objective"].to_numpy(dtype=float)
            ).all()
        )
    else:
        # No successful solve is handled separately by solver_ok / solver_abort_ok.
        objective_ok = True

    finite_ok = finite_state_control_ok and objective_ok

    completion_ok = metrics["completion_pct"] >= MIN_COMPLETION_PCT
    solver_ok = metrics["solver_success_pct"] >= MIN_SOLVER_SUCCESS_PCT
    cte_mean_ok = metrics["mean_cte_post_m"] <= MAX_MEAN_CTE_POST_M
    cte_peak_ok = metrics["max_cte_post_m"] <= MAX_CTE_POST_M
    heading_ok = metrics["rmse_heading_post_rad"] <= MAX_RMSE_HEADING_POST_RAD

    # Use the actual signed logged values and a small floating-point tolerance.
    # This avoids classifying e.g. vx = 0.150000025 m/s as a physical violation
    # of a 0.15 m/s bound.
    input_ok = bool(
        (log["a_cmd"] >= cfg["u_min"][0] - FEAS_TOL).all()
        and (log["a_cmd"] <= cfg["u_max"][0] + FEAS_TOL).all()
        and (log["delta_cmd"] >= cfg["u_min"][1] - FEAS_TOL).all()
        and (log["delta_cmd"] <= cfg["u_max"][1] + FEAS_TOL).all()
        and (log["du_a"] >= cfg["du_min"][0] - FEAS_TOL).all()
        and (log["du_a"] <= cfg["du_max"][0] + FEAS_TOL).all()
        and (log["du_delta"] >= cfg["du_min"][1] - FEAS_TOL).all()
        and (log["du_delta"] <= cfg["du_max"][1] + FEAS_TOL).all()
    )

    vx_domain_ok = bool(
        metrics["min_vx_mps"] >= cfg["vx_min"] - FEAS_TOL
        and metrics["max_vx_mps"] <= cfg["vx_max"] + FEAS_TOL
    )

    solver_abort_ok = not metrics["solver_aborted"]

    metrics["tracking_feasible"] = bool(
        finite_ok and completion_ok and solver_ok and solver_abort_ok
        and cte_mean_ok and cte_peak_ok and heading_ok
        and input_ok and vx_domain_ok
    )
    metrics["host_realtime_feasible"] = bool(
        metrics["deadline_miss_pct_20ms"] <= MAX_HOST_DEADLINE_MISS_PCT
    )

    reasons = []
    checks = {
        "nonfinite": finite_ok,
        "completion": completion_ok,
        "solver": solver_ok,
        "solver_abort": solver_abort_ok,
        "mean_cte": cte_mean_ok,
        "peak_cte": cte_peak_ok,
        "heading": heading_ok,
        "input_bounds": input_ok,
        "vx_domain": vx_domain_ok,
    }
    for name, ok in checks.items():
        if not ok:
            reasons.append(name)
    metrics["infeasible_reasons"] = ";".join(reasons)
    return metrics


def run_case(
    case_id: str,
    Q: np.ndarray,
    R: np.ndarray,
    Qf: np.ndarray,
    save_log=True,
    p_plant_case=None,
    log_dir=None,
    initial_ey_m=None,
    initial_epsi_rad=None,
    sqp_max_iter_override=None,
    enable_solver_fallback=ENABLE_FRESH_ROLLOUT_FALLBACK,
):
    """
    Run one closed-loop NMPC case.

    Important for robustness testing:
      - the NMPC prediction model always uses p_nominal;
      - p_plant_case affects only the simulated plant propagation.
    """
    if p_plant_case is None:
        p_plant_case = p_plant.copy()
    else:
        p_plant_case = np.asarray(p_plant_case, dtype=float).copy()

    if log_dir is None:
        log_dir = CASES_DIR
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    rm = make_reference_manager()

    solver_cfg = cfg.copy()
    if sqp_max_iter_override is not None:
        solver_cfg["sqp_max_iter"] = int(sqp_max_iter_override)

    problem = build_nmpc_solver(
        Fd=Fd_controller,
        N=N,
        Q=Q,
        R=R,
        Qf=Qf,
        cfg=solver_cfg,
        backend="sqpmethod",
    )

    x, u_prev, z_guess, proj0 = build_initial_state_and_guess(
        rm,
        problem,
        initial_ey_m=initial_ey_m,
        initial_epsi_rad=initial_epsi_rad,
    )
    logs = []
    num_steps = int(np.ceil(TMAX / Ts))
    reached_end = False
    solver_fail_count = 0
    consecutive_fail_count = 0
    max_consecutive_fail_count = 0

    for k in range(num_steps):
        t = k * Ts
        ref_result = rm.get_nmpc_reference(x[4], x[5], Ts, N)
        proj = ref_result["projection"]
        xref = ref_result["xref"]

        P = pack_parameters(
            x0=x,
            u_prev=u_prev,
            xref=xref,
            p_model=p_nominal,
        )

        # ------------------------------------------------------------
        # Primary solve: normal shifted warm start
        # ------------------------------------------------------------
        primary_solver_success = False
        primary_solve_ms = 0.0
        primary_return_status = ""
        fallback_attempted = False
        fallback_success = False
        fallback_solve_ms = 0.0
        fallback_return_status = ""

        sol = None
        objective = np.nan

        primary_tic = time.perf_counter()
        try:
            sol_primary = problem["solver"](
                x0=z_guess,
                p=P,
                lbx=problem["lbx"],
                ubx=problem["ubx"],
                lbg=problem["lbg"],
                ubg=problem["ubg"],
            )
            primary_solve_ms = (time.perf_counter() - primary_tic) * 1e3

            primary_stats = problem["solver"].stats()
            primary_solver_success = bool(
                primary_stats.get("success", False)
            )
            primary_return_status = str(
                primary_stats.get("return_status", "")
            )

            if primary_solver_success:
                sol = sol_primary

        except Exception as exc:
            primary_solve_ms = (time.perf_counter() - primary_tic) * 1e3
            primary_solver_success = False
            primary_return_status = (
                f"EXCEPTION: {type(exc).__name__}: {exc}"
            )

        # ------------------------------------------------------------
        # Recovery solve: rebuild a fresh dynamics-consistent guess
        # ------------------------------------------------------------
        if (not primary_solver_success) and enable_solver_fallback:
            fallback_attempted = True

            fresh_guess = build_fresh_dynamics_rollout_guess(
                rm=rm,
                x=x,
                u_prev=u_prev,
                ref_result=ref_result,
            )

            fallback_tic = time.perf_counter()
            try:
                sol_fallback = problem["solver"](
                    x0=fresh_guess,
                    p=P,
                    lbx=problem["lbx"],
                    ubx=problem["ubx"],
                    lbg=problem["lbg"],
                    ubg=problem["ubg"],
                )
                fallback_solve_ms = (
                    time.perf_counter() - fallback_tic
                ) * 1e3

                fallback_stats = problem["solver"].stats()
                fallback_success = bool(
                    fallback_stats.get("success", False)
                )
                fallback_return_status = str(
                    fallback_stats.get("return_status", "")
                )

                if fallback_success:
                    sol = sol_fallback

            except Exception as exc:
                fallback_solve_ms = (
                    time.perf_counter() - fallback_tic
                ) * 1e3
                fallback_success = False
                fallback_return_status = (
                    f"EXCEPTION: {type(exc).__name__}: {exc}"
                )

        # Total optimization time seen by the 20 ms control deadline.
        solve_ms = primary_solve_ms + fallback_solve_ms

        # FINAL solve success after recovery attempt.
        success = bool(
            primary_solver_success or fallback_success
        )

        if success:
            consecutive_fail_count = 0

            X_star, U_star = unpack_solution(sol["x"], N)
            u = U_star[:, 0].copy()
            objective = float(sol["f"])

            # Always warm-start the next sample from the VALID final solution,
            # whether it came from the primary or recovery solve.
            z_guess = shift_warm_start(X_star, U_star)

        else:
            # Count only a FINAL failure.  A recovered primary failure is not
            # treated as a closed-loop solver failure.
            solver_fail_count += 1
            consecutive_fail_count += 1
            max_consecutive_fail_count = max(
                max_consecutive_fail_count,
                consecutive_fail_count,
            )

            # No artificial backup controller: hold the last valid command.
            u = u_prev.copy()

        xref_now = xref[0]
        dpsi = np.arctan2(
            np.sin(x[2] - xref_now[2]),
            np.cos(x[2] - xref_now[2]),
        )
        du = u - u_prev

        logs.append({
            "case_id": case_id,
            "t_s": t,
            "s_progress_m": proj["s"],
            "cross_track_m": proj["distance"],
            "solver_success": int(success),
            "primary_solver_success": int(primary_solver_success),
            "fallback_attempted": int(fallback_attempted),
            "fallback_success": int(fallback_success),
            "solver_fail_streak": int(consecutive_fail_count),
            "primary_solve_ms": primary_solve_ms,
            "fallback_solve_ms": fallback_solve_ms,
            "solve_ms": solve_ms,
            "primary_return_status": primary_return_status,
            "fallback_return_status": fallback_return_status,
            "objective": objective,
            "vx": x[0], "vy": x[1], "psi": x[2], "r": x[3], "X": x[4], "Y": x[5],
            "vx_ref": xref_now[0], "vy_ref": xref_now[1], "psi_ref": xref_now[2],
            "r_ref": xref_now[3], "X_ref": xref_now[4], "Y_ref": xref_now[5],
            "kappa_ref": float(rm.interpolate_curvature(proj["s"])),
            "e_vx": x[0] - xref_now[0],
            "e_vy": x[1] - xref_now[1],
            "e_psi": dpsi,
            "e_r": x[3] - xref_now[3],
            "e_X": x[4] - xref_now[4],
            "e_Y": x[5] - xref_now[5],
            "a_cmd": u[0],
            "delta_cmd": u[1],
            "du_a": du[0],
            "du_delta": du[1],
        })

        # Persistent NMPC failure means this Q/R/Qf case is infeasible.
        # Do NOT continue the plant with an artificial backup controller.
        if consecutive_fail_count >= ABORT_AFTER_CONSECUTIVE_SOLVER_FAILS:
            print(
                f"ABORT: NMPC final solve failed {consecutive_fail_count} "
                f"consecutive samples at t={t:.3f} s"
            )
            break

        x = np.array(Fd_plant(x, u, p_plant_case)).astype(float).reshape(-1)
        u_prev = u.copy()

        # Stop at the REAL CAD endpoint, not at the virtual tail endpoint.
        if proj["s"] >= CAD_S_END - 0.03:
            reached_end = True
            break

        if not np.all(np.isfinite(x)):
            break

    log = pd.DataFrame(logs)
    metrics = compute_metrics(log, reached_end, solver_fail_count)
    metrics["initial_cross_track_m"] = float(proj0["distance"])
    metrics["initial_ey_requested_m"] = (
        np.nan if initial_ey_m is None else float(initial_ey_m)
    )
    metrics["initial_epsi_requested_rad"] = (
        np.nan if initial_epsi_rad is None else float(initial_epsi_rad)
    )
    metrics["sqp_max_iter_used"] = int(solver_cfg["sqp_max_iter"])

    if save_log:
        log.to_csv(log_dir / f"{case_id}.csv", index=False)

    # Release solver memory before the next candidate.
    del problem
    gc.collect()
    return metrics, log


def make_weights(alpha_p, alpha_psi, beta_delta, gamma_f):
    q = Q_CENTER.copy()
    q[2] *= alpha_psi
    q[4] *= alpha_p
    q[5] *= alpha_p

    r = R_CENTER.copy()
    r[1] *= beta_delta

    Q = np.diag(q)
    R = np.diag(r)
    Qf = gamma_f * Q
    return Q, R, Qf


def save_single_case_plots(log, prefix="single"):
    if log is None or log.empty:
        print("WARNING: no simulation samples available; plots were not created.")
        return

    # Real CAD path only; the virtual tail is intentionally hidden.
    plt.figure(figsize=(7.5, 8.5))
    plt.plot(REF_INFO["cad_X"], REF_INFO["cad_Y"], "--", linewidth=1.5, label="CAD reference")
    plt.plot(log["X"], log["Y"], linewidth=1.5, label="NMPC closed-loop")
    plt.axis("equal")
    plt.xlabel("X local [m]")
    plt.ylabel("Y local [m]")
    plt.title("CAD path tracking: NMPC closed-loop")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig_traj = plt.gcf()
    save_figure_safe(fig_traj, f"{prefix}_trajectory.png")
    plt.close(fig_traj)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(log["t_s"], log["cross_track_m"])
    axes[0].set_ylabel("Cross-track [m]")
    axes[0].grid(True)
    axes[1].plot(log["t_s"], log["e_psi"])
    axes[1].set_ylabel("Heading error [rad]")
    axes[1].grid(True)
    axes[2].plot(log["t_s"], log["e_vx"])
    axes[2].set_ylabel("vx error [m/s]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True)
    fig.tight_layout()
    save_figure_safe(fig, f"{prefix}_errors.png")
    plt.close(fig)

    plt.figure(figsize=(9, 4.5))
    plt.plot(log["t_s"], log["solve_ms"])
    plt.axhline(DEADLINE_MS, linestyle="--", label=f"{DEADLINE_MS:.0f} ms sampling deadline")
    plt.xlabel("Time [s]")
    plt.ylabel("Solve time [ms]")
    plt.title("NMPC solver time on development host")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig_solver = plt.gcf()
    save_figure_safe(fig_solver, f"{prefix}_solver_time.png")
    plt.close(fig_solver)


def get_robust_controller_weights():
    Q = np.diag(ROBUST_Q_DIAG)
    R = np.diag(ROBUST_R_DIAG)
    Qf = ROBUST_GAMMA_TERMINAL * Q
    return Q, R, Qf


def run_single():
    import json
    for name,w,size in [('Q_TEST',Q_TEST,6),('R_TEST',R_TEST,2),('QF_TEST',QF_TEST,6)]:
        if w.shape!=(size,) or not np.all(np.isfinite(w)) or np.any(w<=0):
            raise ValueError(f'{name} requires {size} positive finite diagonal entries')
    out=BASE/'single_weight_results';out.mkdir(parents=True,exist_ok=True)
    Q,R,Qf=map(np.diag,[Q_TEST,R_TEST,QF_TEST])
    m,log=run_case('single_custom',Q,R,Qf,save_log=True,log_dir=out,
                   initial_ey_m=.05,initial_epsi_rad=0.)
    pd.DataFrame([m]).to_csv(out/'metrics.csv',index=False)
    config={'Q_INIT':Q_TEST.tolist(),'R_INIT':R_TEST.tolist(),'QF_INIT':QF_TEST.tolist(),
            'Ts':Ts,'N':N,'TMAX':TMAX,'reference':REF_SOURCE.name,'initial_ey':.05,'initial_epsi':0.}
    (out/'weights.json').write_text(json.dumps(config,indent=2))
    fig,ax=plt.subplots(figsize=(7,7))
    ax.plot(log.X_ref,log.Y_ref,'k--',label='Reference')
    ax.plot(log.X,log.Y,label='Fixed-weight NMPC')
    ax.scatter([log.X.iloc[0]],[log.Y.iloc[0]],label='Start')
    ax.axis('equal');ax.set(xlabel='X [m]',ylabel='Y [m]');ax.legend()
    fig.tight_layout();fig.savefig(out/'trajectory.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
    for ax,col,label in zip(axes,['cross_track_m','e_psi','e_vx'],['CTE [m]','Heading error [rad]','Speed error [m/s]']):
        ax.plot(log.t_s,log[col]);ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[-1].set_xlabel('Time [s]');fig.tight_layout()
    fig.savefig(out/'tracking_errors.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    for ax,col,label in zip(axes,['a_cmd','delta_cmd'],['Acceleration [m/s2]','Steering [rad]']):
        ax.plot(log.t_s,log[col]);ax.set_ylabel(label)
    axes[-1].set_xlabel('Time [s]');fig.tight_layout()
    fig.savefig(out/'control_inputs.png',dpi=180);plt.close(fig)
    print('Q=',Q_TEST,'R=',R_TEST,'Qf=',QF_TEST)
    print(pd.DataFrame([m]).to_string(index=False))
    print(f'Results: {out}')


def run_batch(max_cases=None, force_batch=False):
    # Preflight: the weight sweep is meaningful only if the frozen vehicle model
    # can produce at least one reasonable closed-loop response at the center
    # point. If this sanity case fails, investigate model/parameter consistency
    # before interpreting Q/R/Qf results.
    Q0, R0, Qf0 = make_weights(1.0, 1.0, 1.0, 10.0)
    preflight_metrics, _ = run_case(
        "preflight_center", Q0, R0, Qf0, save_log=True
    )
    print("--- PREFLIGHT CENTER CASE ---")
    print(f"tracking_feasible: {preflight_metrics['tracking_feasible']}")
    print(f"solver_success_pct: {preflight_metrics['solver_success_pct']:.2f}")
    print(f"mean_cte_post_m: {preflight_metrics['mean_cte_post_m']:.4f}")
    print(f"max_cte_post_m: {preflight_metrics['max_cte_post_m']:.4f}")
    print(f"rmse_heading_post_rad: {preflight_metrics['rmse_heading_post_rad']:.4f}")
    print(f"infeasible_reasons: {preflight_metrics['infeasible_reasons']}")

    if not preflight_metrics["tracking_feasible"] and not force_batch:
        print("\nBATCH ABORTED BY MODEL-SANITY GUARD.")
        print(
            "The frozen Vehicle Model V1 is not producing a feasible center-case response. "
            "Running 81 weight combinations would mix model-parameter inconsistency with cost tuning."
        )
        print("Use --force-batch only if you intentionally want to explore this inconsistent model.")
        return

    combinations = list(itertools.product(
        ALPHA_POSITION,
        ALPHA_HEADING,
        BETA_STEERING_SMOOTHNESS,
        GAMMA_TERMINAL,
    ))
    if max_cases is not None:
        combinations = combinations[:max_cases]

    summaries = []
    print(f"Running {len(combinations)} Q/R/Qf candidates...")

    for idx, (alpha_p, alpha_psi, beta_delta, gamma_f) in enumerate(combinations, start=1):
        case_id = f"case_{idx:03d}"
        Q, R, Qf = make_weights(alpha_p, alpha_psi, beta_delta, gamma_f)

        print(
            f"[{idx:03d}/{len(combinations):03d}] {case_id} | "
            f"ap={alpha_p:g}, apsi={alpha_psi:g}, bdelta={beta_delta:g}, gf={gamma_f:g}"
        )
        metrics, _ = run_case(case_id, Q, R, Qf, save_log=True)

        row = {
            "case_id": case_id,
            "alpha_position": alpha_p,
            "alpha_heading": alpha_psi,
            "beta_delta": beta_delta,
            "gamma_terminal": gamma_f,
            "q_vx": Q[0, 0],
            "q_vy": Q[1, 1],
            "q_psi": Q[2, 2],
            "q_r": Q[3, 3],
            "q_X": Q[4, 4],
            "q_Y": Q[5, 5],
            "r_da": R[0, 0],
            "r_ddelta": R[1, 1],
            **metrics,
        }
        summaries.append(row)

    summary = pd.DataFrame(summaries)
    summary = summary.sort_values(
        ["tracking_feasible", "mean_cte_post_m", "rmse_heading_post_rad"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    summary.to_csv(RESULTS / "qrqf_sweep_summary.csv", index=False)

    feasible = summary[summary["tracking_feasible"] == True].copy()
    feasible.to_csv(RESULTS / "qrqf_feasible_region.csv", index=False)

    # Ranking inside the feasible region. This is not a new controller cost;
    # it is only a convenient post-processing indicator for inspection.
    if len(feasible):
        feasible["review_score"] = (
            feasible["mean_cte_post_m"] / MAX_MEAN_CTE_POST_M
            + feasible["max_cte_post_m"] / MAX_CTE_POST_M
            + feasible["rmse_heading_post_rad"] / MAX_RMSE_HEADING_POST_RAD
        )
        feasible = feasible.sort_values("review_score").reset_index(drop=True)
        feasible.to_csv(RESULTS / "qrqf_feasible_ranked.csv", index=False)

        # Plot a small set of the best feasible trajectories for visual review.
        plt.figure(figsize=(7.5, 8.5))
        plt.plot(REF_INFO["cad_X"], REF_INFO["cad_Y"], "--", linewidth=1.8, label="CAD reference")
        for case_id in feasible["case_id"].head(10):
            case_log = pd.read_csv(CASES_DIR / f"{case_id}.csv")
            plt.plot(case_log["X"], case_log["Y"], linewidth=0.9, alpha=0.8, label=case_id)
        plt.axis("equal")
        plt.xlabel("X local [m]")
        plt.ylabel("Y local [m]")
        plt.title("Top feasible NMPC trajectories from Q/R/Qf sweep")
        plt.grid(True)
        plt.legend(fontsize=7)
        plt.tight_layout()
        fig_top = plt.gcf()
        save_figure_safe(fig_top, "top_feasible_trajectories.png")
        plt.close(fig_top)

    # Pareto-style inspection plot: geometric tracking vs heading tracking.
    plt.figure(figsize=(8.5, 5.5))
    if len(summary):
        for label, group in summary.groupby("tracking_feasible"):
            plt.scatter(
                group["mean_cte_post_m"],
                group["rmse_heading_post_rad"],
                label="FEASIBLE" if bool(label) else "INFEASIBLE",
                alpha=0.75,
            )
    plt.axvline(MAX_MEAN_CTE_POST_M, linestyle="--")
    plt.axhline(MAX_RMSE_HEADING_POST_RAD, linestyle="--")
    plt.xlabel("Mean cross-track after capture [m]")
    plt.ylabel("Heading RMSE after capture [rad]")
    plt.title("Q/R/Qf sweep: tracking feasibility map")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig_map = plt.gcf()
    save_figure_safe(fig_map, "feasibility_map.png")
    plt.close(fig_map)

    print("\n--- SWEEP COMPLETE ---")
    print(f"Total cases: {len(summary)}")
    print(f"Tracking-feasible cases: {int(summary['tracking_feasible'].sum())}")
    print(f"Results: {RESULTS}")



def build_parameter_robustness_cases():
    """Return 11 OFAT plant-parameter cases: nominal + two sides for 5 parameters."""
    cases = [{
        "case_id": "rob_nominal",
        "parameter": "nominal",
        "perturbation_pct": 0.0,
        "p_plant_case": p_nominal.copy(),
    }]

    for parameter, spec in ROBUST_PARAM_SPECS.items():
        idx = int(spec["index"])
        pct = float(spec["pct"])

        for sign, suffix in [(-1.0, "minus"), (1.0, "plus")]:
            p_case = p_nominal.copy()
            p_case[idx] *= (1.0 + sign * pct)

            case_id = f"rob_{parameter}_{suffix}_{int(round(100*pct)):02d}pct"
            cases.append({
                "case_id": case_id,
                "parameter": parameter,
                "perturbation_pct": 100.0 * sign * pct,
                "p_plant_case": p_case,
            })

    return cases


def _percent_degradation(value, nominal):
    """Signed percentage change; positive means the metric increased."""
    value = float(value)
    nominal = float(nominal)
    if abs(nominal) < 1e-12:
        return np.nan
    return 100.0 * (value - nominal) / nominal


def save_parameter_robustness_plots(summary: pd.DataFrame):
    """Create compact diagnostics for the 11 parameter-mismatch cases."""
    labels = summary["case_id"].tolist()
    x = np.arange(len(summary))

    # Tracking metrics
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    axes[0].bar(x, 100.0 * summary["mean_cte_post_m"].to_numpy(float))
    axes[0].axhline(
        100.0 * MAX_MEAN_CTE_POST_M,
        linestyle="--",
        label="Mean CTE feasibility limit",
    )
    axes[0].set_ylabel("Mean CTE [cm]")
    axes[0].set_title("Parameter robustness: tracking metrics")
    axes[0].grid(True, axis="y")
    axes[0].legend()

    axes[1].bar(x, summary["rmse_heading_post_rad"].to_numpy(float))
    axes[1].axhline(
        MAX_RMSE_HEADING_POST_RAD,
        linestyle="--",
        label="Heading RMSE feasibility limit",
    )
    axes[1].set_ylabel("Heading RMSE [rad]")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].grid(True, axis="y")
    axes[1].legend()

    fig.tight_layout()
    save_figure_safe(fig, "robustness_parameter_metrics.png")
    plt.close(fig)

    # Percentage degradation relative to nominal.
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    axes[0].bar(x, summary["mean_cte_degradation_pct"].to_numpy(float))
    axes[0].axhline(0.0, linewidth=1.0)
    axes[0].set_ylabel("Mean CTE change [%]")
    axes[0].set_title("Parameter robustness: degradation relative to nominal")
    axes[0].grid(True, axis="y")

    axes[1].bar(x, summary["heading_rmse_degradation_pct"].to_numpy(float))
    axes[1].axhline(0.0, linewidth=1.0)
    axes[1].set_ylabel("Heading RMSE change [%]")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].grid(True, axis="y")

    fig.tight_layout()
    save_figure_safe(fig, "robustness_parameter_degradation.png")
    plt.close(fig)

    # Trajectory overlay.
    plt.figure(figsize=(7.5, 8.5))
    plt.plot(
        REF_INFO["cad_X"],
        REF_INFO["cad_Y"],
        "--",
        linewidth=1.8,
        label="CAD reference",
    )

    for case_id in labels:
        path = ROBUST_CASES_DIR / f"{case_id}.csv"
        if path.exists():
            case_log = pd.read_csv(path)
            plt.plot(
                case_log["X"],
                case_log["Y"],
                linewidth=0.9,
                alpha=0.75,
                label=case_id,
            )

    plt.axis("equal")
    plt.xlabel("X local [m]")
    plt.ylabel("Y local [m]")
    plt.title("Parameter robustness: closed-loop trajectories")
    plt.grid(True)
    plt.legend(fontsize=6)
    plt.tight_layout()

    fig = plt.gcf()
    save_figure_safe(fig, "robustness_parameter_trajectories.png")
    plt.close(fig)


def run_parameter_robustness():
    """
    One-factor-at-a-time plant/model mismatch test.

    Controller:
        fixed p_nominal
        fixed Q/R/Qf from selected case_071

    Plant:
        nominal, then ± uncertainty on m, Iz, Cf, Cr, mu_roll.
    """
    Q, R, Qf = get_robust_controller_weights()
    cases = build_parameter_robustness_cases()

    print("--- PARAMETER ROBUSTNESS TEST ---")
    print("Controller model p_nominal =", p_nominal)
    print("Q diag =", np.diag(Q))
    print("R diag =", np.diag(R))
    print(f"Qf/Q multiplier = {ROBUST_GAMMA_TERMINAL:g}")
    print(f"Running {len(cases)} cases...")

    rows = []

    for idx, spec in enumerate(cases, start=1):
        case_id = spec["case_id"]
        parameter = spec["parameter"]
        perturbation_pct = float(spec["perturbation_pct"])
        p_case = np.asarray(spec["p_plant_case"], dtype=float)

        print(
            f"[{idx:02d}/{len(cases):02d}] {case_id} | "
            f"parameter={parameter} | perturbation={perturbation_pct:+.1f}%"
        )

        metrics, _ = run_case(
            case_id,
            Q,
            R,
            Qf,
            save_log=True,
            p_plant_case=p_case,
            log_dir=ROBUST_CASES_DIR,
        )

        row = {
            "case_id": case_id,
            "parameter": parameter,
            "perturbation_pct": perturbation_pct,
            "plant_m_kg": p_case[0],
            "plant_Iz_kgm2": p_case[1],
            "plant_Cf_Nprad": p_case[2],
            "plant_Cr_Nprad": p_case[3],
            "plant_lf_m": p_case[4],
            "plant_lr_m": p_case[5],
            "plant_mu_roll": p_case[6],
            "controller_m_kg": p_nominal[0],
            "controller_Iz_kgm2": p_nominal[1],
            "controller_Cf_Nprad": p_nominal[2],
            "controller_Cr_Nprad": p_nominal[3],
            "controller_mu_roll": p_nominal[6],
            **metrics,
        }
        rows.append(row)

    summary = pd.DataFrame(rows)

    # The nominal case is deliberately first.
    nominal = summary.iloc[0]

    summary["mean_cte_degradation_pct"] = summary["mean_cte_post_m"].apply(
        lambda v: _percent_degradation(v, nominal["mean_cte_post_m"])
    )
    summary["max_cte_degradation_pct"] = summary["max_cte_post_m"].apply(
        lambda v: _percent_degradation(v, nominal["max_cte_post_m"])
    )
    summary["heading_rmse_degradation_pct"] = summary["rmse_heading_post_rad"].apply(
        lambda v: _percent_degradation(v, nominal["rmse_heading_post_rad"])
    )
    summary["vx_rmse_degradation_pct"] = summary["rmse_vx_post_mps"].apply(
        lambda v: _percent_degradation(v, nominal["rmse_vx_post_mps"])
    )

    summary["steering_rate_utilization_pct"] = (
        100.0
        * summary["max_abs_ddelta_rad"].to_numpy(float)
        / float(cfg["du_max"][1])
    )

    out_csv = RESULTS / "robustness_parameter_summary.csv"
    summary.to_csv(out_csv, index=False)

    save_parameter_robustness_plots(summary)

    feasible_count = int(summary["tracking_feasible"].sum())

    print("\n--- PARAMETER ROBUSTNESS COMPLETE ---")
    print(f"Cases: {len(summary)}")
    print(f"Tracking-feasible: {feasible_count}/{len(summary)}")
    print(f"Summary: {out_csv}")
    print(f"Logs: {ROBUST_CASES_DIR}")

    print("\nSensitivity summary:")
    display_cols = [
        "case_id",
        "mean_cte_post_m",
        "max_cte_post_m",
        "rmse_heading_post_rad",
        "rmse_vx_post_mps",
        "mean_cte_degradation_pct",
        "heading_rmse_degradation_pct",
        "max_abs_ddelta_rad",
        "tracking_feasible",
    ]
    print(summary[display_cols].to_string(index=False))


def compute_capture_metrics(log: pd.DataFrame):
    """Return recovery/capture metrics for an initial-condition test."""
    result = {
        "capture_success": False,
        "capture_time_s": np.nan,
        "capture_progress_m": np.nan,
        "capture_index": -1,
        "recovery_max_cte_m": np.nan,
        "recovery_max_abs_heading_rad": np.nan,
    }
    if log is None or log.empty:
        return result

    dwell_samples = max(1, int(np.ceil(CAPTURE_DWELL_S / Ts)))
    cte = log["cross_track_m"].to_numpy(float)
    epsi = np.abs(log["e_psi"].to_numpy(float))
    inside = (cte <= CAPTURE_CTE_M) & (epsi <= CAPTURE_HEADING_RAD)

    run_length = 0
    capture_start_idx = None
    for i, ok in enumerate(inside):
        if ok:
            run_length += 1
            if run_length >= dwell_samples:
                capture_start_idx = i - dwell_samples + 1
                break
        else:
            run_length = 0

    if capture_start_idx is None:
        result["recovery_max_cte_m"] = float(np.max(cte))
        result["recovery_max_abs_heading_rad"] = float(np.max(epsi))
        return result

    result["capture_success"] = True
    result["capture_index"] = int(capture_start_idx)
    result["capture_time_s"] = float(log["t_s"].iloc[capture_start_idx])
    result["capture_progress_m"] = float(log["s_progress_m"].iloc[capture_start_idx])
    recovery = log.iloc[:capture_start_idx + 1]
    result["recovery_max_cte_m"] = float(recovery["cross_track_m"].max())
    result["recovery_max_abs_heading_rad"] = float(np.max(np.abs(recovery["e_psi"].to_numpy(float))))
    return result


def save_initial_robustness_plots(summary: pd.DataFrame):
    ey_values = list(INITIAL_EY_M)
    epsi_values = list(INITIAL_EPSI_DEG)

    feasibility = np.full((len(epsi_values), len(ey_values)), np.nan)
    capture_time = np.full_like(feasibility, np.nan, dtype=float)

    for _, row in summary.iterrows():
        i = epsi_values.index(float(row["initial_epsi_deg"]))
        j = ey_values.index(float(row["initial_ey_m"]))
        feasibility[i, j] = 1.0 if bool(row["initial_condition_feasible"]) else 0.0
        if bool(row["capture_success"]):
            capture_time[i, j] = float(row["capture_time_s"])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(feasibility, origin="lower", aspect="auto",
                   extent=[min(ey_values)-0.025, max(ey_values)+0.025,
                           min(epsi_values)-2.5, max(epsi_values)+2.5],
                   vmin=0.0, vmax=1.0)
    ax.set_xlabel("Initial lateral error e_y [m]")
    ax.set_ylabel("Initial heading error e_psi [deg]")
    ax.set_title("Initial-condition robustness: feasibility map")
    ax.set_xticks(ey_values); ax.set_yticks(epsi_values)
    fig.colorbar(im, ax=ax, label="Feasible = 1, Infeasible = 0")
    fig.tight_layout()
    save_figure_safe(fig, "initial_robustness_feasibility_map.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(capture_time, origin="lower", aspect="auto",
                   extent=[min(ey_values)-0.025, max(ey_values)+0.025,
                           min(epsi_values)-2.5, max(epsi_values)+2.5])
    ax.set_xlabel("Initial lateral error e_y [m]")
    ax.set_ylabel("Initial heading error e_psi [deg]")
    ax.set_title("Initial-condition robustness: capture time")
    ax.set_xticks(ey_values); ax.set_yticks(epsi_values)
    fig.colorbar(im, ax=ax, label="Capture time [s]")
    fig.tight_layout()
    save_figure_safe(fig, "initial_robustness_capture_time.png")
    plt.close(fig)

    plt.figure(figsize=(7.5, 8.5))
    plt.plot(REF_INFO["cad_X"], REF_INFO["cad_Y"], "--", linewidth=1.8, label="CAD reference")
    for case_id in summary["case_id"]:
        path = INITIAL_ROBUST_CASES_DIR / f"{case_id}.csv"
        if path.exists():
            case_log = pd.read_csv(path)
            plt.plot(case_log["X"], case_log["Y"], linewidth=0.7, alpha=0.6)
    plt.axis("equal")
    plt.xlabel("X local [m]"); plt.ylabel("Y local [m]")
    plt.title("Initial-condition robustness: 25 closed-loop trajectories")
    plt.grid(True); plt.tight_layout()
    fig = plt.gcf(); save_figure_safe(fig, "initial_robustness_trajectories.png"); plt.close(fig)


def run_initial_condition_robustness():
    """Run the 5x5 initial lateral/heading error grid with nominal plant/model."""
    Q, R, Qf = get_robust_controller_weights()
    combinations = list(itertools.product(INITIAL_EY_M, INITIAL_EPSI_DEG))

    print("--- INITIAL-CONDITION ROBUSTNESS TEST ---")
    print("Plant/model parameters: nominal")
    print("Q diag =", np.diag(Q))
    print("R diag =", np.diag(R))
    print(f"Qf/Q multiplier = {ROBUST_GAMMA_TERMINAL:g}")
    print(f"Capture corridor: CTE <= {CAPTURE_CTE_M:.3f} m, |heading| <= {np.rad2deg(CAPTURE_HEADING_RAD):.1f} deg, dwell >= {CAPTURE_DWELL_S:.1f} s")
    print(f"Running {len(combinations)} cases...")

    rows = []
    for idx, (ey0, epsi_deg) in enumerate(combinations, start=1):
        epsi_rad = np.deg2rad(epsi_deg)
        ey_tag = f"{ey0:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
        ep_tag = f"{epsi_deg:+.0f}".replace("+", "p").replace("-", "m")
        case_id = f"init_ey_{ey_tag}_epsi_{ep_tag}deg"
        print(f"[{idx:02d}/{len(combinations):02d}] {case_id} | ey0={ey0:+.3f} m | epsi0={epsi_deg:+.1f} deg")

        metrics, log = run_case(
            case_id, Q, R, Qf, save_log=True,
            p_plant_case=p_nominal,
            log_dir=INITIAL_ROBUST_CASES_DIR,
            initial_ey_m=ey0,
            initial_epsi_rad=epsi_rad,
        )
        capture = compute_capture_metrics(log)
        initial_feasible = bool(
            capture["capture_success"]
            and metrics["tracking_feasible"]
            and metrics["reached_end"]
            and not metrics["solver_aborted"]
        )
        rows.append({
            "case_id": case_id,
            "initial_ey_m": float(ey0),
            "initial_epsi_deg": float(epsi_deg),
            "initial_epsi_rad": float(epsi_rad),
            **capture,
            **metrics,
            "initial_condition_feasible": initial_feasible,
        })

    summary = pd.DataFrame(rows).sort_values(["initial_epsi_deg", "initial_ey_m"]).reset_index(drop=True)
    out_csv = RESULTS / "initial_robustness_summary.csv"
    summary.to_csv(out_csv, index=False)
    save_initial_robustness_plots(summary)

    feasible_count = int(summary["initial_condition_feasible"].sum())
    print("\n--- INITIAL-CONDITION ROBUSTNESS COMPLETE ---")
    print(f"Cases: {len(summary)}")
    print(f"Initial-condition feasible: {feasible_count}/{len(summary)}")
    print(f"Summary: {out_csv}")
    print(f"Logs: {INITIAL_ROBUST_CASES_DIR}")
    cols = ["case_id","initial_ey_m","initial_epsi_deg","capture_success","capture_time_s","capture_progress_m","recovery_max_cte_m","recovery_max_abs_heading_rad","mean_cte_post_m","max_cte_post_m","rmse_heading_post_rad","solver_success_pct","initial_condition_feasible"]
    print("\nInitial-condition summary:")
    print(summary[cols].to_string(index=False))



def _solver_failure_diagnostics(log: pd.DataFrame):
    result = {
        "first_solver_fail_time_s": np.nan,
        "first_solver_fail_progress_m": np.nan,
        "last_log_time_s": np.nan,
    }
    if log is None or log.empty:
        return result

    result["last_log_time_s"] = float(log["t_s"].iloc[-1])
    failed = log[log["solver_success"] == 0]
    if not failed.empty:
        result["first_solver_fail_time_s"] = float(failed["t_s"].iloc[0])
        result["first_solver_fail_progress_m"] = float(failed["s_progress_m"].iloc[0])
    return result


def _boundary_failure_mode(metrics, capture):
    if bool(metrics.get("solver_aborted", False)):
        return "solver_abort"
    if not bool(capture.get("capture_success", False)):
        return "no_capture"
    if not bool(metrics.get("tracking_feasible", False)):
        return "tracking_infeasible"
    if not bool(metrics.get("reached_end", False)):
        return "incomplete_route"
    return "feasible"


def save_boundary_plots(summary: pd.DataFrame, diag: pd.DataFrame):
    ey_values = sorted([float(v) for v in BOUNDARY_EY_M])
    epsi_values = sorted([float(v) for v in BOUNDARY_EPSI_DEG])

    feasibility = np.full((len(epsi_values), len(ey_values)), np.nan)
    capture_time = np.full_like(feasibility, np.nan, dtype=float)

    for _, row in summary.iterrows():
        i = epsi_values.index(float(row["initial_epsi_deg"]))
        j = ey_values.index(float(row["initial_ey_m"]))
        feasibility[i, j] = 1.0 if bool(row["boundary_feasible"]) else 0.0
        if bool(row["capture_success"]):
            capture_time[i, j] = float(row["capture_time_s"])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        feasibility,
        origin="lower",
        aspect="auto",
        extent=[min(ey_values)-0.01, max(ey_values)+0.01,
                min(epsi_values)-1.0, max(epsi_values)+1.0],
        vmin=0.0, vmax=1.0,
    )
    ax.set_xlabel("Initial lateral error e_y [m]")
    ax.set_ylabel("Initial heading error e_psi [deg]")
    ax.set_title("Boundary diagnostic: feasible region (SQP max iter = 7)")
    ax.set_xticks(ey_values)
    ax.set_yticks(epsi_values)
    fig.colorbar(im, ax=ax, label="Feasible = 1, Infeasible = 0")
    fig.tight_layout()
    save_figure_safe(fig, "boundary_feasibility_map.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        capture_time,
        origin="lower",
        aspect="auto",
        extent=[min(ey_values)-0.01, max(ey_values)+0.01,
                min(epsi_values)-1.0, max(epsi_values)+1.0],
    )
    ax.set_xlabel("Initial lateral error e_y [m]")
    ax.set_ylabel("Initial heading error e_psi [deg]")
    ax.set_title("Boundary diagnostic: capture time (SQP max iter = 7)")
    ax.set_xticks(ey_values)
    ax.set_yticks(epsi_values)
    fig.colorbar(im, ax=ax, label="Capture time [s]")
    fig.tight_layout()
    save_figure_safe(fig, "boundary_capture_time.png")
    plt.close(fig)

    # Overlay all boundary trajectories.
    plt.figure(figsize=(7.5, 8.5))
    plt.plot(REF_INFO["cad_X"], REF_INFO["cad_Y"], "--", linewidth=1.8, label="CAD reference")
    for case_id in summary["case_id"]:
        path = BOUNDARY_CASES_DIR / f"{case_id}.csv"
        if path.exists():
            case_log = pd.read_csv(path)
            plt.plot(case_log["X"], case_log["Y"], linewidth=0.75, alpha=0.6)
    plt.axis("equal")
    plt.xlabel("X local [m]")
    plt.ylabel("Y local [m]")
    plt.title("Boundary diagnostic: 25 closed-loop trajectories")
    plt.grid(True)
    plt.tight_layout()
    fig = plt.gcf()
    save_figure_safe(fig, "boundary_trajectories.png")
    plt.close(fig)

    # Direct 7-vs-15 SQP iteration diagnostic at (+0.10 m, -10 deg).
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    for _, row in diag.iterrows():
        case_id = row["case_id"]
        path = BOUNDARY_CASES_DIR / f"{case_id}.csv"
        if not path.exists():
            continue
        log = pd.read_csv(path)
        label = f"max_iter={int(row['sqp_max_iter'])}"
        axes[0].plot(log["t_s"], log["cross_track_m"], label=label)
        axes[1].plot(log["t_s"], np.rad2deg(np.abs(log["e_psi"])), label=label)

    axes[0].axhline(CAPTURE_CTE_M, linestyle="--", label="capture CTE threshold")
    axes[0].set_ylabel("Cross-track [m]")
    axes[0].set_title("Solver diagnostic at e_y=+0.10 m, e_psi=-10 deg")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].axhline(np.rad2deg(CAPTURE_HEADING_RAD), linestyle="--", label="capture heading threshold")
    axes[1].set_ylabel("|Heading error| [deg]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True)
    axes[1].legend()
    fig.tight_layout()
    save_figure_safe(fig, "boundary_solver_7_vs_15.png")
    plt.close(fig)


def run_boundary_diagnostic():
    """
    Local 25-case initial-condition scan around the one failed point, followed
    by a controlled solver diagnostic at (+0.10 m, -10 deg) using SQP max
    iterations 7 and 15. Q/R/Qf and plant parameters stay frozen.
    """
    Q, R, Qf = get_robust_controller_weights()

    combinations = list(itertools.product(BOUNDARY_EY_M, BOUNDARY_EPSI_DEG))
    rows = []

    print("--- LOCAL BOUNDARY DIAGNOSTIC ---")
    print("Controller/plant parameters: nominal")
    print("Q diag =", np.diag(Q))
    print("R diag =", np.diag(R))
    print(f"Qf/Q multiplier = {ROBUST_GAMMA_TERMINAL:g}")
    print(f"Grid cases: {len(combinations)} | SQP max iter = {cfg['sqp_max_iter']}")

    for idx, (ey0, epsi_deg) in enumerate(combinations, start=1):
        epsi_rad = np.deg2rad(epsi_deg)
        ey_tag = f"{ey0:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
        ep_tag = f"{epsi_deg:+.0f}".replace("+", "p").replace("-", "m")
        case_id = f"boundary_ey_{ey_tag}_epsi_{ep_tag}deg"

        print(f"[{idx:02d}/{len(combinations):02d}] {case_id} | ey0={ey0:+.3f} m | epsi0={epsi_deg:+.1f} deg")

        metrics, log = run_case(
            case_id, Q, R, Qf,
            save_log=True,
            p_plant_case=p_nominal,
            log_dir=BOUNDARY_CASES_DIR,
            initial_ey_m=ey0,
            initial_epsi_rad=epsi_rad,
            sqp_max_iter_override=7,
        )
        capture = compute_capture_metrics(log)
        solver_diag = _solver_failure_diagnostics(log)
        feasible = bool(
            capture["capture_success"]
            and metrics["tracking_feasible"]
            and metrics["reached_end"]
            and not metrics["solver_aborted"]
        )

        rows.append({
            "case_id": case_id,
            "initial_ey_m": float(ey0),
            "initial_epsi_deg": float(epsi_deg),
            "sqp_max_iter": 7,
            **capture,
            **solver_diag,
            **metrics,
            "failure_mode": _boundary_failure_mode(metrics, capture),
            "boundary_feasible": feasible,
        })

    summary = pd.DataFrame(rows).sort_values(
        ["initial_epsi_deg", "initial_ey_m"]
    ).reset_index(drop=True)
    summary_path = RESULTS / "boundary_robustness_summary.csv"
    summary.to_csv(summary_path, index=False)

    # --------------------------------------------------------
    # Isolated solver convergence diagnostic: 7 vs 15 SQP iters
    # --------------------------------------------------------
    diag_rows = []
    ey0 = BOUNDARY_DIAGNOSTIC_EY_M
    epsi_deg = BOUNDARY_DIAGNOSTIC_EPSI_DEG
    epsi_rad = np.deg2rad(epsi_deg)

    print("\n--- SQP ITERATION DIAGNOSTIC ---")
    for max_iter in BOUNDARY_DIAGNOSTIC_SQP_ITERS:
        case_id = f"solver_diag_iter_{int(max_iter):02d}"
        print(f"{case_id}: ey0={ey0:+.3f} m, epsi0={epsi_deg:+.1f} deg, sqp_max_iter={max_iter}")

        metrics, log = run_case(
            case_id, Q, R, Qf,
            save_log=True,
            p_plant_case=p_nominal,
            log_dir=BOUNDARY_CASES_DIR,
            initial_ey_m=ey0,
            initial_epsi_rad=epsi_rad,
            sqp_max_iter_override=max_iter,
            enable_solver_fallback=False,
        )
        capture = compute_capture_metrics(log)
        solver_diag = _solver_failure_diagnostics(log)
        feasible = bool(
            capture["capture_success"]
            and metrics["tracking_feasible"]
            and metrics["reached_end"]
            and not metrics["solver_aborted"]
        )
        diag_rows.append({
            "case_id": case_id,
            "initial_ey_m": ey0,
            "initial_epsi_deg": epsi_deg,
            "sqp_max_iter": int(max_iter),
            **capture,
            **solver_diag,
            **metrics,
            "failure_mode": _boundary_failure_mode(metrics, capture),
            "diagnostic_feasible": feasible,
        })

    diag = pd.DataFrame(diag_rows)
    diag_path = RESULTS / "boundary_solver_diagnostic.csv"
    diag.to_csv(diag_path, index=False)

    save_boundary_plots(summary, diag)

    print("\n--- BOUNDARY DIAGNOSTIC COMPLETE ---")
    print(f"Grid feasible: {int(summary['boundary_feasible'].sum())}/{len(summary)}")
    print(f"Grid summary: {summary_path}")
    print(f"Solver diagnostic: {diag_path}")
    print(f"Logs: {BOUNDARY_CASES_DIR}")

    cols = [
        "case_id", "initial_ey_m", "initial_epsi_deg",
        "capture_success", "capture_time_s", "solver_success_pct",
        "max_solver_fail_streak", "solver_aborted", "failure_mode",
        "boundary_feasible",
    ]
    print("\nBoundary summary:")
    print(summary[cols].to_string(index=False))

    diag_cols = [
        "case_id", "sqp_max_iter", "capture_success", "capture_time_s",
        "solver_success_pct", "solver_fail_count", "max_solver_fail_streak",
        "solver_aborted", "reached_end", "failure_mode", "diagnostic_feasible",
    ]
    print("\n7-vs-15 SQP diagnostic:")
    print(diag[diag_cols].to_string(index=False))


def _rate_limit_guess(U_guess: np.ndarray, u_prev: np.ndarray):
    """Project an input guess sequentially onto input and per-sample rate bounds."""
    U = np.asarray(U_guess, dtype=float).copy()
    prev = np.asarray(u_prev, dtype=float).reshape(2)

    for i in range(U.shape[1]):
        lo_rate = prev + cfg["du_min"]
        hi_rate = prev + cfg["du_max"]
        lo = np.maximum(cfg["u_min"], lo_rate)
        hi = np.minimum(cfg["u_max"], hi_rate)
        U[:, i] = np.clip(U[:, i], lo, hi)
        prev = U[:, i].copy()

    return U


def _make_diagnostic_guess(rm, x, u_prev, ref_result, mode: str):
    """
    Build one NLP initial guess at the SAME current state/reference.

    Modes
    -----
    shifted_warm_start:
        handled outside this helper because it comes from the previous NMPC solution.
    fresh_dynamics_rollout:
        feedforward U + dynamics-consistent state rollout from current x.
    reference_state_guess:
        X guess is Xref_1...Xref_N; U uses feedforward.
    heading_corrected_rollout:
        feedforward steering plus a decaying heading-error correction, followed by
        a dynamics-consistent state rollout.
    """
    xref = ref_result["xref"]
    s_h = ref_result["s_horizon"]

    kappa_h = rm.interpolate_curvature(s_h[:-1])
    L = p_nominal[4] + p_nominal[5]

    delta_ff = np.arctan(L * kappa_h)
    delta_ff = np.clip(delta_ff, cfg["u_min"][1], cfg["u_max"][1])

    a_ff = np.full(N, p_nominal[6] * p_nominal[7], dtype=float)
    U_guess = np.vstack([a_ff, delta_ff])

    if mode == "heading_corrected_rollout":
        epsi0 = np.arctan2(
            np.sin(x[2] - xref[0, 2]),
            np.cos(x[2] - xref[0, 2]),
        )

        # Positive correction for a negative heading error and vice versa.
        # Decay the correction through the short prediction horizon.
        decay = np.linspace(1.0, 0.25, N)
        U_guess[1, :] += (
            -SOLVER_DIAG_HEADING_GAIN * epsi0 * decay
        )

    U_guess = _rate_limit_guess(U_guess, u_prev)

    if mode == "reference_state_guess":
        X_guess = xref[1:].T.copy()

    elif mode in ("fresh_dynamics_rollout", "heading_corrected_rollout"):
        X_guess = np.zeros((6, N), dtype=float)
        xg = np.asarray(x, dtype=float).copy()

        for i in range(N):
            xg = np.array(
                Fd_controller(xg, U_guess[:, i], p_nominal)
            ).astype(float).reshape(-1)
            X_guess[:, i] = xg

    else:
        raise ValueError(f"Unsupported diagnostic guess mode: {mode}")

    z = np.concatenate([
        X_guess.reshape(-1, order="F"),
        U_guess.reshape(-1, order="F"),
    ])
    return z, X_guess, U_guess


def _max_box_violation(values, lower, upper):
    values = np.asarray(values, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)

    low_v = np.maximum(lower - values, 0.0)
    high_v = np.maximum(values - upper, 0.0)
    return float(max(np.max(low_v), np.max(high_v)))


def _solve_one_diagnostic_nlp(problem, z0, P):
    """Solve one frozen NLP instance and return numerical diagnostics."""
    tic = time.perf_counter()

    try:
        sol = problem["solver"](
            x0=z0,
            p=P,
            lbx=problem["lbx"],
            ubx=problem["ubx"],
            lbg=problem["lbg"],
            ubg=problem["ubg"],
        )
        solve_ms = (time.perf_counter() - tic) * 1e3
        stats = problem["solver"].stats()
        success = bool(stats.get("success", False))
        return_status = str(stats.get("return_status", ""))

        z_star = np.asarray(sol["x"], dtype=float).reshape(-1)
        g_star = np.asarray(sol["g"], dtype=float).reshape(-1)

        X_star, U_star = unpack_solution(z_star, N)

        return {
            "success": success,
            "return_status": return_status,
            "solve_ms": float(solve_ms),
            "objective": float(sol["f"]),
            "max_constraint_violation": _max_box_violation(
                g_star, problem["lbg"], problem["ubg"]
            ),
            "max_decision_bound_violation": _max_box_violation(
                z_star, problem["lbx"], problem["ubx"]
            ),
            "u0_a": float(U_star[0, 0]),
            "u0_delta": float(U_star[1, 0]),
            "pred_min_vx": float(np.min(X_star[0, :])),
            "pred_max_vx": float(np.max(X_star[0, :])),
            "X_star": X_star,
            "U_star": U_star,
        }

    except Exception as exc:
        solve_ms = (time.perf_counter() - tic) * 1e3
        return {
            "success": False,
            "return_status": f"EXCEPTION: {type(exc).__name__}: {exc}",
            "solve_ms": float(solve_ms),
            "objective": np.nan,
            "max_constraint_violation": np.nan,
            "max_decision_bound_violation": np.nan,
            "u0_a": np.nan,
            "u0_delta": np.nan,
            "pred_min_vx": np.nan,
            "pred_max_vx": np.nan,
            "X_star": None,
            "U_star": None,
        }


def run_solver_warmstart_diagnostic():
    """
    Stop the baseline SQP simulation at its FIRST failed solve for
    (e_y0, e_psi0) = (+0.10 m, -10 deg), freeze that exact NLP, and solve it
    repeatedly with different initial guesses.

    This distinguishes:
      1) shifted-warm-start pathology,
      2) SQP sensitivity to initialization,
      3) NLP infeasibility / local difficulty, via an offline IPOPT solve.
    """
    Q, R, Qf = get_robust_controller_weights()

    # --------------------------------------------------------
    # A) Reproduce the failure and freeze the exact failed NLP
    # --------------------------------------------------------
    rm = make_reference_manager()

    sqp_cfg = cfg.copy()
    sqp_cfg["sqp_max_iter"] = int(SOLVER_DIAG_SQP_MAX_ITER)

    baseline_problem = build_nmpc_solver(
        Fd=Fd_controller,
        N=N,
        Q=Q,
        R=R,
        Qf=Qf,
        cfg=sqp_cfg,
        backend="sqpmethod",
    )

    x, u_prev, z_guess, proj0 = build_initial_state_and_guess(
        rm,
        baseline_problem,
        initial_ey_m=SOLVER_DIAG_EY_M,
        initial_epsi_rad=np.deg2rad(SOLVER_DIAG_EPSI_DEG),
    )

    failure_snapshot = None
    last_valid_X = None
    last_valid_U = None

    print("--- SOLVER / WARM-START DIAGNOSTIC ---")
    print(
        f"Target initial condition: ey0={SOLVER_DIAG_EY_M:+.3f} m, "
        f"epsi0={SOLVER_DIAG_EPSI_DEG:+.1f} deg"
    )
    print(
        f"Baseline: sqpmethod + qrqp, max_iter={SOLVER_DIAG_SQP_MAX_ITER}"
    )

    max_steps_to_find_failure = min(int(np.ceil(5.0 / Ts)), int(np.ceil(TMAX / Ts)))

    for k in range(max_steps_to_find_failure):
        t = k * Ts
        ref_result = rm.get_nmpc_reference(x[4], x[5], Ts, N)
        xref = ref_result["xref"]

        P = pack_parameters(
            x0=x,
            u_prev=u_prev,
            xref=xref,
            p_model=p_nominal,
        )

        result = _solve_one_diagnostic_nlp(
            baseline_problem, z_guess, P
        )

        if not result["success"]:
            failure_snapshot = {
                "t_s": float(t),
                "x": x.copy(),
                "u_prev": u_prev.copy(),
                "P": P.copy(),
                "ref_result": ref_result,
                "failed_shifted_guess": z_guess.copy(),
                "baseline_status": result["return_status"],
                "baseline_solve_ms": result["solve_ms"],
            }
            print(
                f"First SQP failure reproduced at t={t:.3f} s | "
                f"status={result['return_status']}"
            )
            break

        X_star = result["X_star"]
        U_star = result["U_star"]
        last_valid_X = X_star.copy()
        last_valid_U = U_star.copy()

        u = U_star[:, 0].copy()
        z_guess = shift_warm_start(X_star, U_star)

        x = np.array(
            Fd_plant(x, u, p_nominal)
        ).astype(float).reshape(-1)
        u_prev = u.copy()

    if failure_snapshot is None:
        raise RuntimeError(
            "The expected SQP failure was not reproduced within the first 5 s. "
            "Do not interpret the diagnostic until the baseline failure is reproduced."
        )

    x_fail = failure_snapshot["x"]
    u_prev_fail = failure_snapshot["u_prev"]
    P_fail = failure_snapshot["P"]
    ref_fail = failure_snapshot["ref_result"]

    # Save the frozen physical/NLP state for traceability.
    snapshot_row = {
        "failure_time_s": failure_snapshot["t_s"],
        "initial_ey_m": SOLVER_DIAG_EY_M,
        "initial_epsi_deg": SOLVER_DIAG_EPSI_DEG,
        "baseline_status": failure_snapshot["baseline_status"],
        "baseline_solve_ms": failure_snapshot["baseline_solve_ms"],
        "vx": x_fail[0],
        "vy": x_fail[1],
        "psi": x_fail[2],
        "r": x_fail[3],
        "X": x_fail[4],
        "Y": x_fail[5],
        "u_prev_a": u_prev_fail[0],
        "u_prev_delta": u_prev_fail[1],
        "xref_vx": ref_fail["xref"][0, 0],
        "xref_vy": ref_fail["xref"][0, 1],
        "xref_psi": ref_fail["xref"][0, 2],
        "xref_r": ref_fail["xref"][0, 3],
        "xref_X": ref_fail["xref"][0, 4],
        "xref_Y": ref_fail["xref"][0, 5],
    }
    pd.DataFrame([snapshot_row]).to_csv(
        SOLVER_DIAG_DIR / "solver_failure_snapshot.csv",
        index=False,
    )

    # --------------------------------------------------------
    # B) Build the three requested SQP guesses at the SAME NLP
    # --------------------------------------------------------
    guesses = {}

    guesses["shifted_warm_start"] = (
        failure_snapshot["failed_shifted_guess"].copy()
    )

    for mode in [
        "fresh_dynamics_rollout",
        "reference_state_guess",
        "heading_corrected_rollout",
    ]:
        z0, Xg, Ug = _make_diagnostic_guess(
            rm, x_fail, u_prev_fail, ref_fail, mode
        )
        guesses[mode] = z0

    # --------------------------------------------------------
    # C) Solve the frozen NLP with independent fresh SQP solvers
    # --------------------------------------------------------
    rows = []
    solutions_for_plot = {}

    for mode, z0 in guesses.items():
        problem = build_nmpc_solver(
            Fd=Fd_controller,
            N=N,
            Q=Q,
            R=R,
            Qf=Qf,
            cfg=sqp_cfg,
            backend="sqpmethod",
        )

        result = _solve_one_diagnostic_nlp(problem, z0, P_fail)

        rows.append({
            "solver": "sqpmethod_qrqp",
            "guess_mode": mode,
            "max_iter": SOLVER_DIAG_SQP_MAX_ITER,
            "failure_time_s": failure_snapshot["t_s"],
            "success": result["success"],
            "return_status": result["return_status"],
            "solve_ms": result["solve_ms"],
            "objective": result["objective"],
            "max_constraint_violation": result["max_constraint_violation"],
            "max_decision_bound_violation": result["max_decision_bound_violation"],
            "u0_a": result["u0_a"],
            "u0_delta": result["u0_delta"],
            "pred_min_vx": result["pred_min_vx"],
            "pred_max_vx": result["pred_max_vx"],
        })

        if result["X_star"] is not None:
            solutions_for_plot[f"SQP: {mode}"] = result["X_star"]

        del problem
        gc.collect()

    # --------------------------------------------------------
    # D) Offline IPOPT feasibility diagnostic on exactly the same NLP
    # --------------------------------------------------------
    ipopt_cfg = cfg.copy()
    ipopt_cfg["ipopt_max_iter"] = int(SOLVER_DIAG_IPOPT_MAX_ITER)

    # Use the dynamics-consistent fresh guess for the primary IPOPT diagnostic.
    z_ipopt = guesses["fresh_dynamics_rollout"]

    ipopt_problem = build_nmpc_solver(
        Fd=Fd_controller,
        N=N,
        Q=Q,
        R=R,
        Qf=Qf,
        cfg=ipopt_cfg,
        backend="ipopt",
    )

    ipopt_result = _solve_one_diagnostic_nlp(
        ipopt_problem, z_ipopt, P_fail
    )

    rows.append({
        "solver": "ipopt",
        "guess_mode": "fresh_dynamics_rollout",
        "max_iter": SOLVER_DIAG_IPOPT_MAX_ITER,
        "failure_time_s": failure_snapshot["t_s"],
        "success": ipopt_result["success"],
        "return_status": ipopt_result["return_status"],
        "solve_ms": ipopt_result["solve_ms"],
        "objective": ipopt_result["objective"],
        "max_constraint_violation": ipopt_result["max_constraint_violation"],
        "max_decision_bound_violation": ipopt_result["max_decision_bound_violation"],
        "u0_a": ipopt_result["u0_a"],
        "u0_delta": ipopt_result["u0_delta"],
        "pred_min_vx": ipopt_result["pred_min_vx"],
        "pred_max_vx": ipopt_result["pred_max_vx"],
    })

    if ipopt_result["X_star"] is not None:
        solutions_for_plot["IPOPT: fresh dynamics"] = ipopt_result["X_star"]

    del ipopt_problem
    del baseline_problem
    gc.collect()

    diag = pd.DataFrame(rows)
    diag_path = SOLVER_DIAG_DIR / "solver_warmstart_diagnostic.csv"
    diag.to_csv(diag_path, index=False)

    # --------------------------------------------------------
    # E) Plot predicted trajectories from successful solves
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    xref = ref_fail["xref"]
    ax.plot(
        xref[:, 4], xref[:, 5], "--",
        linewidth=1.8, label="Reference horizon"
    )
    ax.scatter(
        [x_fail[4]], [x_fail[5]],
        s=50, label="Frozen current state"
    )

    for label, X_star in solutions_for_plot.items():
        ax.plot(
            X_star[4, :], X_star[5, :],
            marker="o", markersize=3, linewidth=1.0,
            label=label,
        )

    ax.axis("equal")
    ax.set_xlabel("X local [m]")
    ax.set_ylabel("Y local [m]")
    ax.set_title(
        "Frozen failed NLP: solver/initial-guess diagnostic"
    )
    ax.grid(True)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure_safe(
        fig, "solver_warmstart_diagnostic.png"
    )
    plt.close(fig)

    print("\n--- FROZEN NLP DIAGNOSTIC COMPLETE ---")
    print(f"Failure snapshot: {SOLVER_DIAG_DIR / 'solver_failure_snapshot.csv'}")
    print(f"Diagnostic table: {diag_path}")
    print(
        diag[
            [
                "solver", "guess_mode", "max_iter", "success",
                "return_status", "solve_ms", "objective",
                "max_constraint_violation",
                "max_decision_bound_violation",
                "u0_a", "u0_delta", "pred_min_vx", "pred_max_vx",
            ]
        ].to_string(index=False)
    )



def run_fallback_validation():
    """
    Full closed-loop A/B validation at the previously isolated failed point:
        e_y0 = +0.10 m, e_psi0 = -10 deg

    The only difference between the two runs is the fresh-rollout retry.
    """
    Q, R, Qf = get_robust_controller_weights()

    rows = []
    logs = {}

    for enabled in [False, True]:
        tag = "fallback_on" if enabled else "fallback_off"
        case_id = f"validation_{tag}"

        print(
            f"Running {case_id}: "
            f"ey0={SOLVER_DIAG_EY_M:+.3f} m, "
            f"epsi0={SOLVER_DIAG_EPSI_DEG:+.1f} deg"
        )

        metrics, log = run_case(
            case_id=case_id,
            Q=Q,
            R=R,
            Qf=Qf,
            save_log=True,
            p_plant_case=p_nominal,
            log_dir=SOLVER_DIAG_DIR,
            initial_ey_m=SOLVER_DIAG_EY_M,
            initial_epsi_rad=np.deg2rad(SOLVER_DIAG_EPSI_DEG),
            sqp_max_iter_override=SOLVER_DIAG_SQP_MAX_ITER,
            enable_solver_fallback=enabled,
        )

        capture = compute_capture_metrics(log)

        rows.append({
            "case_id": case_id,
            "fallback_enabled": bool(enabled),
            **capture,
            **metrics,
        })
        logs[tag] = log

    summary = pd.DataFrame(rows)
    out_csv = SOLVER_DIAG_DIR / "fallback_validation.csv"
    summary.to_csv(out_csv, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    for tag, log in logs.items():
        axes[0].plot(
            log["t_s"],
            log["cross_track_m"],
            label=tag,
        )
        axes[1].plot(
            log["t_s"],
            log["solve_ms"],
            label=tag,
        )

    axes[0].axhline(
        CAPTURE_CTE_M,
        linestyle="--",
        label="capture CTE threshold",
    )
    axes[0].set_ylabel("Cross-track [m]")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].axhline(
        DEADLINE_MS,
        linestyle="--",
        label=f"{DEADLINE_MS:.0f} ms deadline",
    )
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Total solve time [ms]")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    save_figure_safe(fig, "fallback_validation.png")
    plt.close(fig)

    cols = [
        "case_id",
        "fallback_enabled",
        "capture_success",
        "capture_time_s",
        "completion_pct",
        "solver_success_pct",
        "primary_solver_success_pct",
        "fallback_attempt_count",
        "fallback_success_count",
        "fallback_recovery_pct",
        "solver_aborted",
        "mean_cte_post_m",
        "rmse_heading_post_rad",
        "solve_mean_ms",
        "solve_p99_ms",
        "solve_max_ms",
        "deadline_miss_pct_20ms",
        "tracking_feasible",
    ]

    print("\n--- FALLBACK VALIDATION ---")
    print(summary[cols].to_string(index=False))
    print(f"Saved: {out_csv}")



def build_combined_plant_parameters(spec):
    """Create the perturbed simulated-plant parameter vector."""
    p = p_nominal.copy()
    p[0] *= float(spec["m_scale"])
    p[2] *= float(spec["Cf_scale"])
    p[3] *= float(spec["Cr_scale"])
    p[6] *= float(spec["mu_scale"])
    return p


def _safe_pct_change(value, baseline):
    value = float(value)
    baseline = float(baseline)
    if not np.isfinite(value) or not np.isfinite(baseline) or abs(baseline) < 1e-12:
        return np.nan
    return 100.0 * (value - baseline) / baseline


def save_combined_robustness_plots(summary: pd.DataFrame):
    """
    Save compact 4x4 maps plus trajectory overlays.

    Rows    : plant scenarios
    Columns : initial-condition scenarios
    """
    plant_names = [s["name"] for s in COMBINED_PLANT_SCENARIOS]
    initial_names = [s["name"] for s in COMBINED_INITIAL_SCENARIOS]

    feasibility = np.full(
        (len(plant_names), len(initial_names)),
        np.nan,
        dtype=float,
    )
    capture_time = np.full_like(feasibility, np.nan)
    mean_cte_cm = np.full_like(feasibility, np.nan)

    for _, row in summary.iterrows():
        i = plant_names.index(str(row["plant_scenario"]))
        j = initial_names.index(str(row["initial_scenario"]))

        feasibility[i, j] = (
            1.0 if bool(row["combined_feasible"]) else 0.0
        )

        if bool(row["capture_success"]):
            capture_time[i, j] = float(row["capture_time_s"])

        mean_cte_cm[i, j] = 100.0 * float(row["mean_cte_post_m"])

    # Feasibility map
    fig, ax = plt.subplots(figsize=(9, 5.8))
    im = ax.imshow(
        feasibility,
        origin="upper",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xticks(np.arange(len(initial_names)))
    ax.set_xticklabels(initial_names, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(plant_names)))
    ax.set_yticklabels(plant_names)
    ax.set_xlabel("Initial-condition scenario")
    ax.set_ylabel("Plant/model-mismatch scenario")
    ax.set_title("Combined robustness: feasibility map")
    fig.colorbar(im, ax=ax, label="Feasible = 1, Infeasible = 0")
    fig.tight_layout()
    save_figure_safe(fig, "combined_robustness_feasibility_map.png")
    plt.close(fig)

    # Capture time map
    fig, ax = plt.subplots(figsize=(9, 5.8))
    im = ax.imshow(
        capture_time,
        origin="upper",
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(initial_names)))
    ax.set_xticklabels(initial_names, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(plant_names)))
    ax.set_yticklabels(plant_names)
    ax.set_xlabel("Initial-condition scenario")
    ax.set_ylabel("Plant/model-mismatch scenario")
    ax.set_title("Combined robustness: capture time [s]")
    fig.colorbar(im, ax=ax, label="Capture time [s]")
    fig.tight_layout()
    save_figure_safe(fig, "combined_robustness_capture_time.png")
    plt.close(fig)

    # Mean post-capture CTE map
    fig, ax = plt.subplots(figsize=(9, 5.8))
    im = ax.imshow(
        mean_cte_cm,
        origin="upper",
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(initial_names)))
    ax.set_xticklabels(initial_names, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(plant_names)))
    ax.set_yticklabels(plant_names)
    ax.set_xlabel("Initial-condition scenario")
    ax.set_ylabel("Plant/model-mismatch scenario")
    ax.set_title("Combined robustness: mean post-capture CTE [cm]")
    fig.colorbar(im, ax=ax, label="Mean CTE [cm]")
    fig.tight_layout()
    save_figure_safe(fig, "combined_robustness_mean_cte.png")
    plt.close(fig)

    # One overlay for all 16 closed-loop trajectories.
    plt.figure(figsize=(7.5, 8.5))
    plt.plot(
        REF_INFO["cad_X"],
        REF_INFO["cad_Y"],
        "--",
        linewidth=1.8,
        label="CAD reference",
    )

    for case_id in summary["case_id"]:
        path = COMBINED_CASES_DIR / f"{case_id}.csv"
        if not path.exists():
            continue
        log = pd.read_csv(path)
        plt.plot(
            log["X"],
            log["Y"],
            linewidth=0.75,
            alpha=0.65,
        )

    plt.axis("equal")
    plt.xlabel("X local [m]")
    plt.ylabel("Y local [m]")
    plt.title("Combined robustness: 16 closed-loop trajectories")
    plt.grid(True)
    plt.tight_layout()
    fig = plt.gcf()
    save_figure_safe(fig, "combined_robustness_trajectories.png")
    plt.close(fig)


def run_combined_robustness():
    """
    Combined parameter-mismatch + initial-condition robustness study.

    Fixed controller:
        Q, R, Qf from selected case_071
        prediction model = p_nominal
        shifted warm start + fresh dynamics fallback

    Varied simulated plant:
        nominal
        mu +25%
        Cf +20% and Cr +20%
        m -10%, Cf +20%, Cr +20%, mu +25%

    Varied initial condition:
        baseline      (+0.05 m,   0 deg)
        moderate      (+0.05 m,  -5 deg)
        strong        (+0.10 m, -10 deg)
        strong_mirror (-0.10 m, +10 deg)
    """
    Q, R, Qf = get_robust_controller_weights()

    total_cases = (
        len(COMBINED_PLANT_SCENARIOS)
        * len(COMBINED_INITIAL_SCENARIOS)
    )

    print("--- COMBINED ROBUSTNESS TEST ---")
    print("Controller model: fixed p_nominal")
    print("Fresh-rollout fallback: ENABLED")
    print("Q diag =", np.diag(Q))
    print("R diag =", np.diag(R))
    print(f"Qf/Q multiplier = {ROBUST_GAMMA_TERMINAL:g}")
    print(f"Cases: {total_cases}")

    rows = []
    case_counter = 0

    for plant_spec in COMBINED_PLANT_SCENARIOS:
        p_case = build_combined_plant_parameters(plant_spec)

        for init_spec in COMBINED_INITIAL_SCENARIOS:
            case_counter += 1

            plant_name = str(plant_spec["name"])
            init_name = str(init_spec["name"])
            ey0 = float(init_spec["ey_m"])
            epsi_deg = float(init_spec["epsi_deg"])
            epsi_rad = np.deg2rad(epsi_deg)

            case_id = (
                f"comb_{case_counter:02d}_"
                f"{plant_name}_{init_name}"
            )

            print(
                f"[{case_counter:02d}/{total_cases:02d}] {case_id} | "
                f"ey0={ey0:+.3f} m | "
                f"epsi0={epsi_deg:+.1f} deg | "
                f"m={plant_spec['m_scale']:.2f}x | "
                f"Cf={plant_spec['Cf_scale']:.2f}x | "
                f"Cr={plant_spec['Cr_scale']:.2f}x | "
                f"mu={plant_spec['mu_scale']:.2f}x"
            )

            metrics, log = run_case(
                case_id=case_id,
                Q=Q,
                R=R,
                Qf=Qf,
                save_log=True,
                p_plant_case=p_case,
                log_dir=COMBINED_CASES_DIR,
                initial_ey_m=ey0,
                initial_epsi_rad=epsi_rad,
                sqp_max_iter_override=cfg["sqp_max_iter"],
                enable_solver_fallback=True,
            )

            capture = compute_capture_metrics(log)

            combined_feasible = bool(
                capture["capture_success"]
                and metrics["tracking_feasible"]
                and metrics["reached_end"]
                and not metrics["solver_aborted"]
            )

            rows.append({
                "case_id": case_id,
                "plant_scenario": plant_name,
                "initial_scenario": init_name,
                "initial_ey_m": ey0,
                "initial_epsi_deg": epsi_deg,

                "m_scale": float(plant_spec["m_scale"]),
                "Cf_scale": float(plant_spec["Cf_scale"]),
                "Cr_scale": float(plant_spec["Cr_scale"]),
                "mu_scale": float(plant_spec["mu_scale"]),

                "plant_m_kg": float(p_case[0]),
                "plant_Iz_kgm2": float(p_case[1]),
                "plant_Cf_Nprad": float(p_case[2]),
                "plant_Cr_Nprad": float(p_case[3]),
                "plant_mu_roll": float(p_case[6]),

                **capture,
                **metrics,
                "combined_feasible": combined_feasible,
            })

    summary = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Degradation is referenced to the NOMINAL PLANT with the
    # SAME initial-condition scenario, not to one global case.
    # --------------------------------------------------------
    summary["mean_cte_degradation_pct"] = np.nan
    summary["heading_rmse_degradation_pct"] = np.nan
    summary["vx_rmse_degradation_pct"] = np.nan
    summary["capture_time_delta_s"] = np.nan

    for init_name in summary["initial_scenario"].unique():
        mask = summary["initial_scenario"] == init_name
        group = summary.loc[mask]

        base = group[
            group["plant_scenario"] == "plant_nominal"
        ]

        if len(base) != 1:
            continue

        base = base.iloc[0]

        summary.loc[mask, "mean_cte_degradation_pct"] = (
            group["mean_cte_post_m"].apply(
                lambda v: _safe_pct_change(
                    v, base["mean_cte_post_m"]
                )
            ).to_numpy()
        )

        summary.loc[mask, "heading_rmse_degradation_pct"] = (
            group["rmse_heading_post_rad"].apply(
                lambda v: _safe_pct_change(
                    v, base["rmse_heading_post_rad"]
                )
            ).to_numpy()
        )

        summary.loc[mask, "vx_rmse_degradation_pct"] = (
            group["rmse_vx_post_mps"].apply(
                lambda v: _safe_pct_change(
                    v, base["rmse_vx_post_mps"]
                )
            ).to_numpy()
        )

        if bool(base["capture_success"]):
            summary.loc[mask, "capture_time_delta_s"] = (
                group["capture_time_s"].to_numpy(float)
                - float(base["capture_time_s"])
            )

    # Stable output ordering.
    plant_order = {
        s["name"]: i
        for i, s in enumerate(COMBINED_PLANT_SCENARIOS)
    }
    initial_order = {
        s["name"]: i
        for i, s in enumerate(COMBINED_INITIAL_SCENARIOS)
    }

    summary["_plant_order"] = summary["plant_scenario"].map(plant_order)
    summary["_initial_order"] = summary["initial_scenario"].map(initial_order)

    summary = summary.sort_values(
        ["_plant_order", "_initial_order"]
    ).drop(
        columns=["_plant_order", "_initial_order"]
    ).reset_index(drop=True)

    out_csv = RESULTS / "combined_robustness_summary.csv"
    summary.to_csv(out_csv, index=False)

    # Convenience subset: failed cases only.
    failed = summary[summary["combined_feasible"] == False].copy()
    failed.to_csv(
        RESULTS / "combined_robustness_failed_cases.csv",
        index=False,
    )

    save_combined_robustness_plots(summary)

    feasible_count = int(summary["combined_feasible"].sum())

    print("\n--- COMBINED ROBUSTNESS COMPLETE ---")
    print(
        f"Combined feasible: "
        f"{feasible_count}/{len(summary)}"
    )
    print(f"Summary: {out_csv}")
    print(f"Logs: {COMBINED_CASES_DIR}")

    display_cols = [
        "case_id",
        "plant_scenario",
        "initial_scenario",
        "capture_success",
        "capture_time_s",
        "completion_pct",
        "solver_success_pct",
        "primary_solver_success_pct",
        "fallback_attempt_count",
        "fallback_success_count",
        "mean_cte_post_m",
        "max_cte_post_m",
        "rmse_heading_post_rad",
        "mean_cte_degradation_pct",
        "heading_rmse_degradation_pct",
        "solve_p99_ms",
        "deadline_miss_pct_20ms",
        "tracking_feasible",
        "combined_feasible",
    ]

    print("\nCombined robustness summary:")
    print(summary[display_cols].to_string(index=False))



def make_qrqf_region_weights(alpha_p, alpha_psi, beta_delta, gamma_f):
    """
    Exact Q/R/Qf parameterization used in the tuning sweep that produced case_071.

    Base:
        Q0 = diag(12, 3, 15, 6, 105, 110)
        R0 = diag(0.6, 8)

    Scaling:
        q_psi *= alpha_psi
        q_X, q_Y *= alpha_p
        r_ddelta *= beta_delta
        Qf = gamma_f * Q
    """
    q = QRQF_BASE_Q_DIAG.copy()
    q[2] *= float(alpha_psi)
    q[4] *= float(alpha_p)
    q[5] *= float(alpha_p)

    r = QRQF_BASE_R_DIAG.copy()
    r[1] *= float(beta_delta)

    Q = np.diag(q)
    R = np.diag(r)
    Qf = float(gamma_f) * Q

    return Q, R, Qf


def _qrqf_case_table():
    """Return the deterministic 81-case table in original itertools.product order."""
    combinations = list(itertools.product(
        ALPHA_POSITION,
        ALPHA_HEADING,
        BETA_STEERING_SMOOTHNESS,
        GAMMA_TERMINAL,
    ))

    rows = []
    for idx, (ap, apsi, bdelta, gf) in enumerate(combinations, start=1):
        rows.append({
            "case_id": f"case_{idx:03d}",
            "alpha_position": float(ap),
            "alpha_heading": float(apsi),
            "beta_delta": float(bdelta),
            "gamma_terminal": float(gf),
        })
    return pd.DataFrame(rows)


def _qrqf_load_reference():
    ref = pd.read_csv(REF_SOURCE)
    needed = ["s_m", "X_ref_m", "Y_ref_m", "psi_ref_rad"]
    missing = [c for c in needed if c not in ref.columns]
    if missing:
        raise ValueError(f"Reference CSV missing columns: {missing}")

    ref = (
        ref.sort_values("s_m")
        .drop_duplicates("s_m", keep="first")
        .reset_index(drop=True)
    )

    # Real CAD section only.
    ref = ref[ref["s_m"] <= CAD_S_END + 1e-9].copy()
    return ref


def _qrqf_read_log(case_id):
    path = QRQF_REGION_CASES_DIR / f"{case_id}.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    log = pd.read_csv(path)

    needed = ["s_progress_m", "X", "Y"]
    missing = [c for c in needed if c not in log.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    log = log[np.isfinite(log["s_progress_m"])].copy()
    log = log.sort_values("s_progress_m")
    log = log.drop_duplicates("s_progress_m", keep="last")
    return log


def _qrqf_bool_series(series):
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def _qrqf_build_envelope(summary, ref):
    """
    Resample all feasible trajectories on one common progress grid and compute
    a signed lateral-error tube relative to the reference centerline.
    """
    feasible = summary[_qrqf_bool_series(summary["tracking_feasible"])].copy()

    if feasible.empty:
        raise RuntimeError("No tracking-feasible Q/R/Qf cases exist.")

    logs = {}
    for case_id in feasible["case_id"].astype(str):
        log = _qrqf_read_log(case_id)
        if len(log) >= 2:
            logs[case_id] = log

    if not logs:
        raise RuntimeError("No usable Q/R/Qf trajectory logs found.")

    s_start = max(
        float(ref["s_m"].min()),
        max(float(log["s_progress_m"].min()) for log in logs.values()),
    )
    s_end = min(
        float(ref["s_m"].max()),
        min(float(log["s_progress_m"].max()) for log in logs.values()),
    )

    if s_end <= s_start:
        raise RuntimeError("No common path-progress interval across feasible cases.")

    s_grid = np.linspace(s_start, s_end, 1400)

    s_ref = ref["s_m"].to_numpy(float)
    x_ref = np.interp(s_grid, s_ref, ref["X_ref_m"].to_numpy(float))
    y_ref = np.interp(s_grid, s_ref, ref["Y_ref_m"].to_numpy(float))
    psi_ref = np.interp(
        s_grid,
        s_ref,
        np.unwrap(ref["psi_ref_rad"].to_numpy(float)),
    )

    ey_rows = []
    xy = {}

    for case_id, log in logs.items():
        s = log["s_progress_m"].to_numpy(float)
        X = np.interp(s_grid, s, log["X"].to_numpy(float))
        Y = np.interp(s_grid, s, log["Y"].to_numpy(float))

        dx = X - x_ref
        dy = Y - y_ref
        ey = -dx * np.sin(psi_ref) + dy * np.cos(psi_ref)

        ey_rows.append(ey)
        xy[case_id] = (X, Y)

    ey_mat = np.vstack(ey_rows)

    stats = pd.DataFrame({
        "s_m": s_grid,
        "X_ref_m": x_ref,
        "Y_ref_m": y_ref,
        "psi_ref_rad": psi_ref,
        "ey_min_m": np.min(ey_mat, axis=0),
        "ey_p05_m": np.percentile(ey_mat, 5, axis=0),
        "ey_mean_m": np.mean(ey_mat, axis=0),
        "ey_p95_m": np.percentile(ey_mat, 95, axis=0),
        "ey_max_m": np.max(ey_mat, axis=0),
    })

    stats["full_band_width_m"] = (
        stats["ey_max_m"] - stats["ey_min_m"]
    )
    stats["p90_band_width_m"] = (
        stats["ey_p95_m"] - stats["ey_p05_m"]
    )

    return feasible, stats, xy


def _qrqf_offset_curve(stats, column):
    ey = stats[column].to_numpy(float)
    psi = stats["psi_ref_rad"].to_numpy(float)
    xr = stats["X_ref_m"].to_numpy(float)
    yr = stats["Y_ref_m"].to_numpy(float)

    X = xr - ey * np.sin(psi)
    Y = yr + ey * np.cos(psi)
    return X, Y


def save_qrqf_region_plots(summary):
    ref = _qrqf_load_reference()
    feasible, stats, xy = _qrqf_build_envelope(summary, ref)

    stats_path = RESULTS / "qrqf_trajectory_region_stats.csv"
    stats.to_csv(stats_path, index=False)

    # --------------------------------------------------------
    # A) Full trajectory region
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 9.0))

    for case_id in feasible["case_id"].astype(str):
        X, Y = xy[case_id]
        ax.plot(X, Y, linewidth=0.45, alpha=0.18)

    min_curve = _qrqf_offset_curve(stats, "ey_min_m")
    max_curve = _qrqf_offset_curve(stats, "ey_max_m")
    p05_curve = _qrqf_offset_curve(stats, "ey_p05_m")
    p95_curve = _qrqf_offset_curve(stats, "ey_p95_m")

    xpoly = np.concatenate([min_curve[0], max_curve[0][::-1]])
    ypoly = np.concatenate([min_curve[1], max_curve[1][::-1]])
    ax.fill(
        xpoly, ypoly,
        alpha=0.16,
        label="Full feasible Q/R/Qf trajectory region",
    )

    xpoly90 = np.concatenate([p05_curve[0], p95_curve[0][::-1]])
    ypoly90 = np.concatenate([p05_curve[1], p95_curve[1][::-1]])
    ax.fill(
        xpoly90, ypoly90,
        alpha=0.26,
        label="5-95% trajectory region",
    )

    ax.plot(
        ref["X_ref_m"],
        ref["Y_ref_m"],
        "--",
        linewidth=2.0,
        label="CAD reference",
    )

    if QRQF_SELECTED_CASE in xy:
        X, Y = xy[QRQF_SELECTED_CASE]
        ax.plot(
            X, Y,
            linewidth=2.2,
            label=f"Selected controller ({QRQF_SELECTED_CASE})",
        )

    ax.axis("equal")
    ax.set_xlabel("X local [m]")
    ax.set_ylabel("Y local [m]")
    ax.set_title("NMPC trajectory region under Q/R/Qf variation")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = RESULTS / "qrqf_trajectory_region.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # B) Signed lateral envelope versus progress
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    s = stats["s_m"].to_numpy(float)
    ax.fill_between(
        s,
        100.0 * stats["ey_min_m"].to_numpy(float),
        100.0 * stats["ey_max_m"].to_numpy(float),
        alpha=0.17,
        label="Full min-max region",
    )
    ax.fill_between(
        s,
        100.0 * stats["ey_p05_m"].to_numpy(float),
        100.0 * stats["ey_p95_m"].to_numpy(float),
        alpha=0.28,
        label="5-95% region",
    )
    ax.plot(
        s,
        100.0 * stats["ey_mean_m"].to_numpy(float),
        linewidth=1.5,
        label="Mean signed lateral error",
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.1, label="Reference centerline")
    ax.set_xlabel("Path progress s [m]")
    ax.set_ylabel("Signed lateral deviation [cm]")
    ax.set_title("Q/R/Qf sweep: lateral trajectory envelope")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    path2 = RESULTS / "qrqf_lateral_envelope.png"
    fig.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # C) One-factor plots around case_071
    # --------------------------------------------------------
    selected_row = summary[
        summary["case_id"].astype(str) == QRQF_SELECTED_CASE
    ]

    if selected_row.empty:
        raise RuntimeError(
            f"{QRQF_SELECTED_CASE} not found in Q/R/Qf summary."
        )

    selected_row = selected_row.iloc[0]

    factors = [
        ("alpha_position", "Q position-weight scale", "qrqf_effect_Q_position.png"),
        ("alpha_heading", "Q heading-weight scale", "qrqf_effect_Q_heading.png"),
        ("beta_delta", "R steering-smoothness scale", "qrqf_effect_R_steering.png"),
        ("gamma_terminal", "Qf/Q terminal multiplier", "qrqf_effect_Qf_terminal.png"),
    ]

    factor_rows = []

    for varied, title, filename in factors:
        mask = np.ones(len(summary), dtype=bool)

        for fixed in [
            "alpha_position",
            "alpha_heading",
            "beta_delta",
            "gamma_terminal",
        ]:
            if fixed == varied:
                continue

            mask &= np.isclose(
                summary[fixed].to_numpy(float),
                float(selected_row[fixed]),
                atol=1e-12,
                rtol=0.0,
            )

        group = summary.loc[mask].sort_values(varied)

        fig, ax = plt.subplots(figsize=(8.5, 9.0))
        ax.plot(
            ref["X_ref_m"],
            ref["Y_ref_m"],
            "--",
            linewidth=2.0,
            label="CAD reference",
        )

        for _, row in group.iterrows():
            case_id = str(row["case_id"])
            log = _qrqf_read_log(case_id)

            feasible_flag = bool(row["tracking_feasible"])
            label = (
                f"{varied}={float(row[varied]):g} | "
                f"{case_id} | "
                f"{'feasible' if feasible_flag else 'infeasible'}"
            )

            ax.plot(
                log["X"],
                log["Y"],
                linewidth=1.45,
                alpha=0.90,
                label=label,
            )

            factor_rows.append({
                "varied_factor": varied,
                "factor_value": float(row[varied]),
                "case_id": case_id,
                "tracking_feasible": feasible_flag,
                "alpha_position": float(row["alpha_position"]),
                "alpha_heading": float(row["alpha_heading"]),
                "beta_delta": float(row["beta_delta"]),
                "gamma_terminal": float(row["gamma_terminal"]),
                "mean_cte_post_m": float(row["mean_cte_post_m"]),
                "max_cte_post_m": float(row["max_cte_post_m"]),
                "rmse_heading_post_rad": float(row["rmse_heading_post_rad"]),
            })

        ax.axis("equal")
        ax.set_xlabel("X local [m]")
        ax.set_ylabel("Y local [m]")
        ax.set_title(f"Isolated sensitivity: {title}")
        ax.grid(True)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(
            RESULTS / filename,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    pd.DataFrame(factor_rows).to_csv(
        RESULTS / "qrqf_one_factor_cases.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------
    width = stats["full_band_width_m"].to_numpy(float)
    p90 = stats["p90_band_width_m"].to_numpy(float)
    imax = int(np.argmax(width))

    print("\n--- Q/R/Qf TRAJECTORY REGION ---")
    print(f"Feasible trajectories used: {len(feasible)}/{len(summary)}")
    print(f"Selected controller: {QRQF_SELECTED_CASE}")
    print(
        f"Common s interval: "
        f"{stats['s_m'].iloc[0]:.3f} -> {stats['s_m'].iloc[-1]:.3f} m"
    )
    print(
        f"Maximum full-band width: "
        f"{100.0 * width[imax]:.3f} cm "
        f"at s={stats['s_m'].iloc[imax]:.3f} m"
    )
    print(
        f"Mean full-band width: "
        f"{100.0 * np.mean(width):.3f} cm"
    )
    print(
        f"Mean 5-95% band width: "
        f"{100.0 * np.mean(p90):.3f} cm"
    )

    print("\nSaved:")
    print(RESULTS / "qrqf_trajectory_region.png")
    print(RESULTS / "qrqf_lateral_envelope.png")
    print(RESULTS / "qrqf_effect_Q_position.png")
    print(RESULTS / "qrqf_effect_Q_heading.png")
    print(RESULTS / "qrqf_effect_R_steering.png")
    print(RESULTS / "qrqf_effect_Qf_terminal.png")
    print(stats_path)


def run_qrqf_region_rebuild(max_cases=None):
    """
    Re-run the exact Q/R/Qf sweep used to select case_071, save all trajectories,
    then construct the empirical trajectory region.

    For scientific reproducibility the fresh-rollout recovery is disabled here:
    this recreates the original tuning experiment rather than the later
    robustness/recovery experiment.
    """
    combinations = list(itertools.product(
        ALPHA_POSITION,
        ALPHA_HEADING,
        BETA_STEERING_SMOOTHNESS,
        GAMMA_TERMINAL,
    ))

    if max_cases is not None:
        combinations = combinations[:int(max_cases)]

    print("--- REBUILD Q/R/Qf TRAJECTORY REGION ---")
    print("Exact tuning base:")
    print("Q0 diag =", QRQF_BASE_Q_DIAG)
    print("R0 diag =", QRQF_BASE_R_DIAG)
    print("Fresh-rollout fallback: DISABLED for tuning reproducibility")
    print(f"Cases to run: {len(combinations)}")

    summaries = []

    for idx, (ap, apsi, bdelta, gf) in enumerate(combinations, start=1):
        case_id = f"case_{idx:03d}"

        Q, R, Qf = make_qrqf_region_weights(
            ap, apsi, bdelta, gf
        )

        print(
            f"[{idx:03d}/{len(combinations):03d}] {case_id} | "
            f"ap={ap:g}, apsi={apsi:g}, "
            f"bdelta={bdelta:g}, gf={gf:g}"
        )

        metrics, _ = run_case(
            case_id=case_id,
            Q=Q,
            R=R,
            Qf=Qf,
            save_log=True,
            p_plant_case=p_nominal,
            log_dir=QRQF_REGION_CASES_DIR,
            initial_ey_m=None,
            initial_epsi_rad=None,
            sqp_max_iter_override=cfg["sqp_max_iter"],
            enable_solver_fallback=False,
        )

        summaries.append({
            "case_id": case_id,
            "alpha_position": float(ap),
            "alpha_heading": float(apsi),
            "beta_delta": float(bdelta),
            "gamma_terminal": float(gf),

            "q_vx": float(Q[0, 0]),
            "q_vy": float(Q[1, 1]),
            "q_psi": float(Q[2, 2]),
            "q_r": float(Q[3, 3]),
            "q_X": float(Q[4, 4]),
            "q_Y": float(Q[5, 5]),

            "r_da": float(R[0, 0]),
            "r_ddelta": float(R[1, 1]),

            **metrics,
        })

    summary = pd.DataFrame(summaries)

    summary = summary.sort_values(
        [
            "tracking_feasible",
            "mean_cte_post_m",
            "rmse_heading_post_rad",
        ],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    out_csv = RESULTS / "qrqf_region_summary.csv"
    summary.to_csv(out_csv, index=False)

    feasible = summary[
        summary["tracking_feasible"] == True
    ].copy()

    feasible.to_csv(
        RESULTS / "qrqf_region_feasible.csv",
        index=False,
    )

    print("\n--- SWEEP COMPLETE ---")
    print(f"Cases run: {len(summary)}")
    print(
        f"Tracking feasible: "
        f"{int(summary['tracking_feasible'].sum())}/{len(summary)}"
    )
    print("Summary:", out_csv)

    if len(summary) == 81:
        selected = summary[
            summary["case_id"] == QRQF_SELECTED_CASE
        ]
        if not selected.empty:
            row = selected.iloc[0]
            print("\ncase_071 check:")
            print(
                "  alpha_position =", row["alpha_position"],
                "| alpha_heading =", row["alpha_heading"],
                "| beta_delta =", row["beta_delta"],
                "| gamma_terminal =", row["gamma_terminal"],
            )
            print(
                "  Q diag =",
                [
                    row["q_vx"], row["q_vy"], row["q_psi"],
                    row["q_r"], row["q_X"], row["q_Y"],
                ],
            )
            print(
                "  R diag =",
                [row["r_da"], row["r_ddelta"]],
            )

        save_qrqf_region_plots(summary)
    else:
        print(
            "\nSmoke-test only: plots are skipped because the complete "
            "81-case grid has not been run."
        )




def _stress_signed_lateral_error(log):
    """Signed lateral error in the local Frenet normal direction."""
    eX = log["e_X"].to_numpy(float)
    eY = log["e_Y"].to_numpy(float)
    psi_ref = log["psi_ref"].to_numpy(float)
    return -eX * np.sin(psi_ref) + eY * np.cos(psi_ref)


def _stress_case_label(factor, factor_value, multiplier, feasible):
    state = "feasible" if feasible else "infeasible"
    return f"{factor}={factor_value:g} ({multiplier:g}x) | {state}"


def _stress_weights_from_factor(factor, multiplier):
    vals = QRQF_STRESS_SELECTED.copy()
    vals[factor] = float(vals[factor]) * float(multiplier)

    Q, R, Qf = make_qrqf_region_weights(
        vals["alpha_position"],
        vals["alpha_heading"],
        vals["beta_delta"],
        vals["gamma_terminal"],
    )
    return vals, Q, R, Qf


def _save_qrqf_stress_factor_plots(factor, rows, logs):
    """
    Save three plots for one isolated factor:
      1) XY trajectories,
      2) signed lateral error versus path progress,
      3) control/heading response versus path progress.
    """
    ref = _qrqf_load_reference()

    # --------------------------------------------------------
    # A) Full XY trajectory
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 9.0))
    ax.plot(
        ref["X_ref_m"],
        ref["Y_ref_m"],
        "--",
        linewidth=2.0,
        label="CAD reference",
    )

    for row in rows:
        case_id = row["case_id"]
        log = logs[case_id]
        ax.plot(
            log["X"],
            log["Y"],
            linewidth=1.7,
            label=_stress_case_label(
                factor,
                row["factor_value"],
                row["multiplier"],
                row["tracking_feasible"],
            ),
        )

    ax.axis("equal")
    ax.set_xlabel("X local [m]")
    ax.set_ylabel("Y local [m]")
    ax.set_title(f"Extended sensitivity: {factor}")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure_safe(fig, f"qrqf_stress_{factor}_trajectory.png", dpi=200)
    plt.close(fig)

    # --------------------------------------------------------
    # B) Signed lateral error vs path progress
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    for row in rows:
        case_id = row["case_id"]
        log = logs[case_id]
        ey = _stress_signed_lateral_error(log)

        ax.plot(
            log["s_progress_m"],
            100.0 * ey,
            linewidth=1.6,
            label=_stress_case_label(
                factor,
                row["factor_value"],
                row["multiplier"],
                row["tracking_feasible"],
            ),
        )

    ax.axhline(0.0, linestyle="--", linewidth=1.1, label="Reference centerline")
    ax.set_xlabel("Path progress s [m]")
    ax.set_ylabel("Signed lateral error [cm]")
    ax.set_title(f"Extended sensitivity: {factor} — lateral error")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure_safe(fig, f"qrqf_stress_{factor}_lateral_error.png", dpi=200)
    plt.close(fig)

    # --------------------------------------------------------
    # C) Response quantity most relevant to each factor
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    for row in rows:
        case_id = row["case_id"]
        log = logs[case_id]

        if factor == "beta_delta":
            y = np.rad2deg(log["delta_cmd"].to_numpy(float))
            ylabel = "Steering command δ [deg]"
            title_suffix = "steering command"
        elif factor == "alpha_heading":
            y = np.rad2deg(log["e_psi"].to_numpy(float))
            ylabel = "Heading error [deg]"
            title_suffix = "heading error"
        elif factor == "gamma_terminal":
            y = 100.0 * log["cross_track_m"].to_numpy(float)
            ylabel = "Cross-track error [cm]"
            title_suffix = "cross-track error"
        else:
            y = 100.0 * log["cross_track_m"].to_numpy(float)
            ylabel = "Cross-track error [cm]"
            title_suffix = "cross-track error"

        ax.plot(
            log["s_progress_m"],
            y,
            linewidth=1.5,
            label=_stress_case_label(
                factor,
                row["factor_value"],
                row["multiplier"],
                row["tracking_feasible"],
            ),
        )

    ax.set_xlabel("Path progress s [m]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Extended sensitivity: {factor} — {title_suffix}")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure_safe(fig, f"qrqf_stress_{factor}_response.png", dpi=200)
    plt.close(fig)


def _save_qrqf_stress_metric_plots(summary):
    """
    Metric plots make the trade-offs visible even when XY trajectories overlap.
    """
    factors = [
        "alpha_position",
        "alpha_heading",
        "beta_delta",
        "gamma_terminal",
    ]

    for factor in factors:
        group = summary[summary["varied_factor"] == factor].copy()
        group = group.sort_values("multiplier")

        x = group["factor_value"].to_numpy(float)

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.plot(
            x,
            100.0 * group["mean_cte_post_m"].to_numpy(float),
            marker="o",
            linewidth=1.6,
            label="Mean post CTE [cm]",
        )
        ax.plot(
            x,
            100.0 * group["max_cte_post_m"].to_numpy(float),
            marker="o",
            linewidth=1.6,
            label="Max post CTE [cm]",
        )

        ax.set_xlabel(factor)
        ax.set_ylabel("Position error [cm]")
        ax.set_title(f"Extended sensitivity metrics: {factor}")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        save_figure_safe(fig, f"qrqf_stress_{factor}_cte_metrics.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.plot(
            x,
            np.rad2deg(group["rmse_heading_post_rad"].to_numpy(float)),
            marker="o",
            linewidth=1.6,
            label="Heading RMSE [deg]",
        )

        if factor == "beta_delta":
            ax2 = ax.twinx()
            ax2.plot(
                x,
                np.rad2deg(group["max_abs_ddelta_rad"].to_numpy(float)),
                marker="s",
                linewidth=1.6,
                label="Max |Δδ| [deg/sample]",
            )
            ax2.set_ylabel("Max |Δδ| [deg/sample]")

        ax.set_xlabel(factor)
        ax.set_ylabel("Heading RMSE [deg]")
        ax.set_title(f"Extended sensitivity trade-off: {factor}")
        ax.grid(True)
        fig.tight_layout()
        save_figure_safe(fig, f"qrqf_stress_{factor}_heading_metrics.png", dpi=200)
        plt.close(fig)


def run_qrqf_extended_stress(max_cases=None):
    """
    Deliberately widen Q/R/Qf factors to make their influence visually clear.

    One-factor-at-a-time around case_071:
        multiplier = 1x, 2x, 4x, 8x

    Selected baseline:
        alpha_position = 1.75
        alpha_heading  = 1.0
        beta_delta     = 2.0
        gamma_terminal = 15

    Important:
      - These high values are sensitivity/stress settings, not recommended tuning.
      - The final solver-recovery strategy remains ON to avoid confusing
        numerical warm-start pathology with controller-weight sensitivity.
    """
    factors = [
        "alpha_position",
        "alpha_heading",
        "beta_delta",
        "gamma_terminal",
    ]

    plan = []
    for factor in factors:
        for multiplier in QRQF_STRESS_MULTIPLIERS:
            plan.append((factor, float(multiplier)))

    if max_cases is not None:
        plan = plan[:int(max_cases)]

    print("--- EXTENDED Q/R/Qf SENSITIVITY STRESS TEST ---")
    print("One-factor-at-a-time around case_071")
    print("Multipliers:", QRQF_STRESS_MULTIPLIERS)
    print("Fresh-rollout fallback: ENABLED")
    print(f"Cases to run: {len(plan)}")

    rows = []
    logs = {}

    for idx, (factor, multiplier) in enumerate(plan, start=1):
        vals, Q, R, Qf = _stress_weights_from_factor(
            factor, multiplier
        )

        factor_value = vals[factor]
        case_id = (
            f"stress_{factor}_"
            f"x{str(multiplier).replace('.', 'p')}"
        )

        print(
            f"[{idx:02d}/{len(plan):02d}] {case_id} | "
            f"{factor}={factor_value:g} | multiplier={multiplier:g}x"
        )
        print("  Q diag =", np.diag(Q))
        print("  R diag =", np.diag(R))
        print("  gamma =", vals["gamma_terminal"])

        metrics, log = run_case(
            case_id=case_id,
            Q=Q,
            R=R,
            Qf=Qf,
            save_log=True,
            p_plant_case=p_nominal,
            log_dir=QRQF_STRESS_CASES_DIR,
            initial_ey_m=None,
            initial_epsi_rad=None,
            sqp_max_iter_override=cfg["sqp_max_iter"],
            enable_solver_fallback=True,
        )

        logs[case_id] = log

        rows.append({
            "case_id": case_id,
            "varied_factor": factor,
            "multiplier": multiplier,
            "factor_value": float(factor_value),

            "alpha_position": vals["alpha_position"],
            "alpha_heading": vals["alpha_heading"],
            "beta_delta": vals["beta_delta"],
            "gamma_terminal": vals["gamma_terminal"],

            "q_vx": Q[0, 0],
            "q_vy": Q[1, 1],
            "q_psi": Q[2, 2],
            "q_r": Q[3, 3],
            "q_X": Q[4, 4],
            "q_Y": Q[5, 5],
            "r_da": R[0, 0],
            "r_ddelta": R[1, 1],
            "qf_scale": vals["gamma_terminal"],

            **metrics,
        })

    summary = pd.DataFrame(rows)
    out_csv = RESULTS / "qrqf_extended_stress_summary.csv"
    summary.to_csv(out_csv, index=False)

    # Create plots only for factors that have at least two completed runs.
    for factor in factors:
        factor_rows = [
            row for row in rows
            if row["varied_factor"] == factor
        ]
        if len(factor_rows) < 2:
            continue

        factor_logs = {
            row["case_id"]: logs[row["case_id"]]
            for row in factor_rows
        }
        _save_qrqf_stress_factor_plots(
            factor, factor_rows, factor_logs
        )

    if len(summary) > 0:
        _save_qrqf_stress_metric_plots(summary)

    print("\n--- EXTENDED STRESS COMPLETE ---")
    print("Summary:", out_csv)

    display_cols = [
        "case_id",
        "varied_factor",
        "multiplier",
        "factor_value",
        "mean_cte_post_m",
        "max_cte_post_m",
        "rmse_heading_post_rad",
        "max_abs_ddelta_rad",
        "solver_success_pct",
        "fallback_attempt_count",
        "deadline_miss_pct_20ms",
        "tracking_feasible",
        "infeasible_reasons",
    ]
    print(summary[display_cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "robustness", "initial_robustness", "boundary", "solver_diag", "fallback_validation", "combined_robustness", "qrqf_region", "qrqf_stress"],
        default="single",
        help=(
            "single = frozen nominal controller; "
            "batch = legacy Q/R/Qf sweep; "
            "robustness = 11-case plant-parameter mismatch test; "
            "initial_robustness = 25-case initial-condition grid; "
            "boundary = local RoA/solver boundary diagnostic; "
            "solver_diag = frozen-NLP warm-start + IPOPT diagnostic; "
            "fallback_validation = full-loop fallback OFF/ON comparison; "
            "combined_robustness = 16-case model-mismatch + initial-error stress test; "
            "qrqf_region = rerun exact 81-case Q/R/Qf tuning sweep and draw trajectory region; "
            "qrqf_stress = extended one-factor Q/R/Qf sensitivity (1x/2x/4x/8x)"
        ),
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional batch smoke-test limit, e.g. --max-cases 4",
    )
    parser.add_argument(
        "--force-batch",
        action="store_true",
        help="Run batch even if the frozen-model center-case preflight is infeasible",
    )
    args = parser.parse_args()

    print(f"Reference source: {REF_SOURCE.name}")
    print(
        f"CAD s_end={CAD_S_END:.4f} m | "
        f"virtual tail end={REF_INFO['tail_s_end']:.4f} m | N={N} | Ts={Ts:.3f} s"
    )

    if args.mode == "single":
        run_single()
    elif args.mode == "batch":
        run_batch(args.max_cases, force_batch=args.force_batch)
    elif args.mode == "robustness":
        run_parameter_robustness()
    elif args.mode == "initial_robustness":
        run_initial_condition_robustness()
    elif args.mode == "boundary":
        run_boundary_diagnostic()
    elif args.mode == "solver_diag":
        run_solver_warmstart_diagnostic()
    elif args.mode == "fallback_validation":
        run_fallback_validation()
    elif args.mode == "combined_robustness":
        run_combined_robustness()
    elif args.mode == "qrqf_region":
        run_qrqf_region_rebuild(args.max_cases)
    else:
        run_qrqf_extended_stress(args.max_cases)


if __name__ == "__main__":
    main()
