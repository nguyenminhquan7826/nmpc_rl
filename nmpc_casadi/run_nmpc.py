"""Reproducible fixed-weight nonlinear MPC without Actor, Critic or RL.

Run from nmpc_casadi (PowerShell):
  python run_nmpc.py --output results_nmpc/nominal --seconds 120
  python run_nmpc.py --reference reference_split/train_xref.csv \
      --output results_nmpc/train_path --seconds 120
  python run_nmpc.py --reference reference_split/test_xref.csv \
      --output results_nmpc/test_path --seconds 150

NMPC dynamics, RK4, objectives, constraints and solver are unchanged from
baseline.py. The original fixed nominal controller weights are retained and
are NOT claimed to be the paper's numerical tuning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import baseline as b
from cad_eval_reference import prepare_cad_evaluation_reference, sha256_file


def _plot(log, full_reference, output):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(full_reference["X_ref_m"], full_reference["Y_ref_m"],
            "k--", linewidth=1.8, label="CAD reference")
    ax.plot(log["X"], log["Y"], linewidth=1.5, label="NMPC vehicle")
    ax.set(xlabel="X (m)", ylabel="Y (m)", title="Fixed-weight NMPC path tracking")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "trajectory.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(log["s_progress_m"], 100. * log["cross_track_m"])
    ax.set(xlabel="Path progress (m)", ylabel="CTE (cm)",
           title="Fixed-weight NMPC cross-track error")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "cross_track_error.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(log["t_s"], log["e_psi"], label="Heading error (rad)")
    axes[1].plot(log["t_s"], log["delta_cmd"], label="Steering (rad)")
    axes[1].plot(log["t_s"], log["a_cmd"], label="Acceleration (m/s²)")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.25)
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(output / "heading_and_control.png", dpi=170)
    plt.close(fig)


def _fixed_diagonals(args):
    q, r, qf = b.get_robust_controller_weights()
    d = {
        "Q": np.diag(q) if args.q is None else np.asarray(args.q, dtype=float),
        "R": np.diag(r) if args.r is None else np.asarray(args.r, dtype=float),
        "Qf": np.diag(qf) if args.qf is None else np.asarray(args.qf, dtype=float),
    }
    for name, values in d.items():
        expected = 2 if name == "R" else 6
        if values.shape != (expected,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{name} requires {expected} strictly positive finite entries")
    return np.diag(d["Q"]), np.diag(d["R"]), np.diag(d["Qf"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=None,
                        help="Independent 10-column CAD XREF CSV; otherwise original baseline reference")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=120.)
    parser.add_argument("--ey", type=float, default=.05,
                        help="Initial signed lateral error relative to start tangent, m")
    parser.add_argument("--heading-deg", type=float, default=0.)
    parser.add_argument("--q", type=float, nargs=6, metavar="QI",
                        help="Optional fixed 6 diagonal Q coefficients")
    parser.add_argument("--r", type=float, nargs=2, metavar="RI",
                        help="Optional fixed 2 diagonal R coefficients")
    parser.add_argument("--qf", type=float, nargs=6, metavar="QFI",
                        help="Optional fixed 6 diagonal terminal Qf coefficients")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    if not np.isfinite(args.ey) or not np.isfinite(args.heading_deg):
        parser.error("initial offsets must be finite")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be an empty directory (existing results protected)")

    q, r, qf = _fixed_diagonals(args)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.reference is not None:
        source = args.reference.expanduser().resolve(strict=True)
        ref_info, manifest = prepare_cad_evaluation_reference(
            source, args.output, b.TAIL_LENGTH_M,
            min_speed_mps=float(b.cfg["vx_min"]),
            max_speed_mps=min(float(b.TEST_SPEED_MPS), float(b.cfg["vx_max"])),
            wheelbase_m=float(b.WHEELBASE_M),
            steering_limit_rad=float(min(-b.cfg["u_min"][1], b.cfg["u_max"][1])),
        )
        b.REF_INFO = ref_info
        b.TAIL_FILE = Path(manifest["tail_file"])
        b.CAD_S_END = float(ref_info["cad_s_end"])
        reference_sha = manifest["reference_sha256"]
        ref_label = str(source)
        full_reference = pd.read_csv(source)
    else:
        ref_label = str(b.REF_SOURCE.resolve())
        reference_sha = sha256_file(b.REF_SOURCE)
        full_reference = pd.read_csv(b.REF_SOURCE)

    b.TMAX = args.seconds
    metadata = {
        "controller": "fixed_weight_nmpc_no_rl",
        "reference": ref_label,
        "reference_sha256": reference_sha,
        "reference_length_m": float(b.CAD_S_END),
        "Ts_s": float(b.Ts), "N": int(b.N),
        "initial_ey_m": args.ey, "initial_heading_deg": args.heading_deg,
        "time_limit_s": args.seconds,
        "Q_diagonal": np.diag(q).tolist(),
        "R_diagonal": np.diag(r).tolist(),
        "Qf_diagonal": np.diag(qf).tolist(),
        "dynamics_convention": "restoring (legacy default; pending paper verification)",
        "speed_profile": "reference_csv; legacy default may be fixed at 0.1 m/s",
        "note": "No Actor/Critic; weights are fixed. No claim of strict paper reproduction.",
    }
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2),
                                             encoding="utf-8")
    summary, logs = b.run_case(
        "nmpc_fixed", q, r, qf, save_log=True, log_dir=args.output,
        initial_ey_m=args.ey, initial_epsi_rad=np.deg2rad(args.heading_deg),
        p_plant_case=b.p_nominal,
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)),
        encoding="utf-8",
    )
    if not args.no_plots:
        _plot(logs, full_reference, args.output)
    print(json.dumps({
        "reference": ref_label, "completion_pct": summary.get("completion_pct"),
        "mean_cte_post_m": summary.get("mean_cte_post_m"),
        "solver_success_pct": summary.get("solver_success_pct"),
        "reached_end": summary.get("reached_end"),
        "saved_to": str(args.output),
    }, default=str), flush=True)


if __name__ == "__main__":
    main()
