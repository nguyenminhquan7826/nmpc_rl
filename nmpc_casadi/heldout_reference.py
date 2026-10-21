"""Deterministic held-out open-path geometry for out-of-trajectory NMPC evaluation.

This file is NEVER imported by training. The path was not used to select any
R3A checkpoint. Geometry is newly integrated from analytic curvature, not
transformed/resampled from the training CAD track. Its arc length and speed
are matched to the training task for an interpretable comparison.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


HELDOUT_ID = "heldout_s_bend_v1"
HELDOUT_CURVATURE_FORMULA = (
    "kappa(s) = 1.15*sin(4*pi*s/L) - 0.45*sin(2*pi*s/L) [1/m]"
)


def generate_heldout_s_bend(path, length_m, speed_mps=0.10, ds_target_m=0.025):
    """Write a C2-smooth-heading, bidirectionally curved open reference.

    Curvature starts/ends at zero, changes sign, and remains within the
    nominal steering limit: |atan(wheelbase*kappa)| < hardware steering.
    X,Y are integrated by midpoint heading with ds as the independent
    variable; the curve is not copied from the training reference.
    """
    path = Path(path)
    L = float(length_m)
    v = float(speed_mps)
    ds = float(ds_target_m)
    if not np.isfinite(L) or L <= 2.0:
        raise ValueError("Held-out reference length must exceed 2 m")
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("Reference speed must be positive and finite")
    if not np.isfinite(ds) or ds <= 0.0 or ds > 0.05:
        raise ValueError("Reference waypoint spacing must be in (0, 0.05] m")

    n = max(2, int(np.ceil(L / ds)))
    s = np.linspace(0., L, n + 1, dtype=float)
    kappa = 1.15*np.sin(4.*np.pi*s/L) - 0.45*np.sin(2.*np.pi*s/L)
    delta_s = np.diff(s)
    heading = np.zeros(n + 1, dtype=float)
    heading[1:] = np.cumsum(
        0.5*(kappa[:-1] + kappa[1:])*delta_s
    )
    theta_mid = 0.5*(heading[:-1] + heading[1:])
    X = np.zeros(n + 1, dtype=float)
    Y = np.zeros(n + 1, dtype=float)
    X[1:] = np.cumsum(delta_s*np.cos(theta_mid))
    Y[1:] = np.cumsum(delta_s*np.sin(theta_mid))

    data = pd.DataFrame({
        "index": np.arange(n + 1),
        "s_m": s,
        "t_ref_s": s/v,
        "vx_ref_mps": np.full(n + 1, v),
        "vy_ref_mps": np.zeros(n + 1),
        "psi_ref_rad": heading,
        "r_ref_radps": v*kappa,
        "X_ref_m": X,
        "Y_ref_m": Y,
        "kappa_1_per_m": kappa,
    })
    # Check geometry and kinematic coherence BEFORE writing any output.
    if not np.all(np.isfinite(data.to_numpy(float))):
        raise ValueError("Nonfinite held-out reference")
    if np.any(np.diff(s) <= 0) or abs(s[-1] - L) > 1e-10:
        raise ValueError("Arc-length reference is not strictly increasing")
    if np.max(np.abs(np.diff(heading)/delta_s)) > 1.55:
        raise ValueError("Held-out path exceeds intended curvature envelope")
    if np.max(np.abs(np.arctan(0.2515*kappa))) >= np.deg2rad(33.44):
        raise ValueError("Held-out path exceeds nominal steering angle")
    if not (np.any(kappa > .5) and np.any(kappa < -.5)):
        raise ValueError("Held-out path must turn in both directions")

    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, float_format="%.12g")
    return data


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def describe_geometry(data):
    """Physical shape metadata: no claims of equivalent difficulty."""
    return {
        "number_of_points": int(len(data)),
        "length_m": float(data["s_m"].iloc[-1] - data["s_m"].iloc[0]),
        "reference_speed_mps": float(data["vx_ref_mps"].mean()),
        "max_abs_curvature_1_per_m": float(np.max(
            np.abs(data["kappa_1_per_m"].to_numpy(float))
        )),
        "min_curvature_1_per_m": float(data["kappa_1_per_m"].min()),
        "max_curvature_1_per_m": float(data["kappa_1_per_m"].max()),
        "heading_range_rad": [
            float(data["psi_ref_rad"].min()),
            float(data["psi_ref_rad"].max()),
        ],
        "start_xy_m": [float(data["X_ref_m"].iloc[0]),
                       float(data["Y_ref_m"].iloc[0])],
        "end_xy_m": [float(data["X_ref_m"].iloc[-1]),
                     float(data["Y_ref_m"].iloc[-1])],
    }


def compare_geometry(training_csv, heldout_data):
    """Quantify geometric separation from the training CAD trajectory.

    Training CSV is read for DIAGNOSTICS ONLY. It is never the source used
    to construct heldout_data except for caller-chosen total arc length.
    """
    training = pd.read_csv(training_csv).sort_values("s_m")
    ts = training["s_m"].to_numpy(float)
    s = heldout_data["s_m"].to_numpy(float)
    if np.any(np.diff(ts) <= 0):
        raise ValueError("Training reference arc length must increase")
    txy = np.column_stack([
        np.interp(s, ts, training[key].to_numpy(float))
        for key in ("X_ref_m", "Y_ref_m")
    ])
    xy = heldout_data[["X_ref_m", "Y_ref_m"]].to_numpy(float)
    separation = np.linalg.norm(xy - txy, axis=1)
    tk = np.interp(s, ts, training["kappa_1_per_m"].to_numpy(float))
    hk = heldout_data["kappa_1_per_m"].to_numpy(float)
    curvature_rms_diff = float(np.sqrt(np.mean((tk - hk)**2)))
    geometric_rms_diff = float(np.sqrt(np.mean(separation**2)))
    if curvature_rms_diff < .1 or geometric_rms_diff < .5:
        raise RuntimeError(
            "Held-out path insufficiently different from training path"
        )
    return {
        "rms_xy_separation_from_training_m": geometric_rms_diff,
        "max_xy_separation_from_training_m": float(separation.max()),
        "rms_curvature_difference_1_per_m": curvature_rms_diff,
        "exact_same_waypoints": bool(np.array_equal(txy, xy)),
    }
