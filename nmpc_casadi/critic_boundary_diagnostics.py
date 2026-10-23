"""Non-mutating TRAIN-only boundary diagnostic for direct six-P Critic."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def analyse_run(root, tolerance=1e-6):
    root = Path(root)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    cfg = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    if cfg.get("reference_role") != "TRAIN_ONLY":
        raise ValueError("Only TRAIN_ONLY data allowed")
    lo = np.asarray(cfg["P_lower"], dtype=float)
    hi = np.asarray(cfg["P_upper"], dtype=float)
    if lo.shape != (6,) or hi.shape != (6,) or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)) or np.any(lo <= 0) or np.any(hi <= lo):
        raise ValueError("Expected six positive bounded P parameters")
    df = pd.read_csv(root / "training_history.csv")
    required = ["rms_td", "mean_cte_m"] + [f"qf_final_{i}" for i in range(6)]
    if any(k not in df for k in required):
        raise ValueError("Missing Critic history columns")
    p = df[[f"qf_final_{i}" for i in range(6)]].to_numpy(dtype=float)
    if len(p) == 0 or not np.all(np.isfinite(p)):
        raise ValueError("Empty or nonfinite Critic history")
    if np.any(p < lo - 1e-8) or np.any(p > hi + 1e-8):
        raise ValueError("P values outside reported projection box")
    f = (p - lo) / (hi - lo)
    lower, upper = f <= tolerance, f >= 1 - tolerance
    n = min(10, len(df))
    return {
        "type": "TRAIN_ONLY_PROJECTED_P_BOUNDARY_DIAGNOSTIC",
        "episodes": len(df), "boundary_fraction_tolerance": tolerance,
        "actor_lr": cfg.get("actor_lr"), "critic_lr": cfg.get("critic_lr"),
        "P_lower": lo.tolist(), "P_upper": hi.tolist(),
        "final_P": p[-1].tolist(), "final_normalized_P": f[-1].tolist(),
        "at_lower_episodes_per_component": lower.sum(axis=0).tolist(),
        "at_upper_episodes_per_component": upper.sum(axis=0).tolist(),
        "final_at_lower": lower[-1].tolist(), "final_at_upper": upper[-1].tolist(),
        "episodes_with_any_boundary": int(np.any(lower | upper, axis=1).sum()),
        "rms_td_first10": float(df["rms_td"].iloc[:n].mean()),
        "rms_td_last10": float(df["rms_td"].iloc[-n:].mean()),
        "mean_cte_first10_m": float(df["mean_cte_m"].iloc[:n].mean()),
        "mean_cte_last10_m": float(df["mean_cte_m"].iloc[-n:].mean()),
        "caveat": "Boundary hits and lower TRAIN TD loss do NOT prove convergence, stability or held-out improvement."
    }

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--boundary-fraction-tolerance", type=float, default=1e-6)
    args = ap.parse_args(argv)
    report = analyse_run(args.train_root, args.boundary_fraction_tolerance)
    if args.output.exists():
        ap.error("Output already exists; refusing overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status":"OK", "episodes":report["episodes"], "output":str(args.output)}))

if __name__ == "__main__":
    main()
