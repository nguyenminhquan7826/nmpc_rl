"""Empirical NMPC parameter-box screening on the STRICTLY PINNED TRAIN path.

Usage from nmpc_casadi, where reference_split/train_xref.csv is local:

  python screen_training_weights.py --reference reference_split/train_xref.csv \
      --output results_screen/quick_20pct --margin 0.20 --profile quick

  python screen_training_weights.py --reference reference_split/train_xref.csv \
      --output results_screen/full_20pct --margin 0.20 --profile full

IMPORTANT: Numerical box and sample count are exploratory, not approved
training/stability bounds. This routine does NOT train Actor or Critic,
does NOT access TEST, and cannot prove all points inside the box satisfy
paper terminal-set or recursive-feasibility assumptions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from training_initialization import (
    TRAIN_REFERENCE_SHA256, LEARNING_INITIAL, GOOD_NMPC,
    representative_weights, screening_box,
)


def verify_train_reference(reference):
    path = Path(reference).expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise ValueError("Expected a local TRAIN CSV")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != TRAIN_REFERENCE_SHA256:
        raise ValueError(
            "TRAIN reference SHA256 mismatch; refusing to run on changed "
            f"or held-out reference: expected {TRAIN_REFERENCE_SHA256}, got {digest}"
        )
    return path, digest


def initial_conditions(profile):
    if profile == "quick":
        return ((.05, 0.),)
    if profile == "full":
        # Separate conditions, not assumed to be stable or Markov-certified.
        return ((.05, 0.), (-.05, 0.), (.05, 5.))
    raise ValueError(profile)


def screen_one(reference, root, label, qrp, ey, heading_deg, seconds):
    q, r, p = qrp
    directory = root / f"{label}_ey{ey:+.3f}_hdg{heading_deg:+.1f}"
    cmd = [
        sys.executable, str(Path(__file__).with_name("run_nmpc.py")),
        "--reference", str(reference),
        "--seconds", str(seconds),
        "--ey", str(ey),
        "--heading-deg", str(heading_deg),
        "--q", *(str(v) for v in q),
        "--r", *(str(v) for v in r),
        "--qf", *(str(v) for v in p),
        "--output", str(directory),
        "--no-plots",
    ]
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent,
                            text=True, capture_output=True, check=False)
    record = {
        "label": label, "ey_m": ey, "heading_deg": heading_deg,
        "case_dir": str(directory), "returncode": result.returncode,
        "q": list(map(float, q)), "r": list(map(float, r)),
        "p": list(map(float, p)),
    }
    if result.returncode != 0:
        record.update(status="PROCESS_FAILED",
                      error=(result.stderr or result.stdout)[-2400:])
        return record

    summary_path = directory / "summary.json"
    config_path = directory / "config.json"
    if not summary_path.exists() or not config_path.exists():
        record.update(status="OUTPUT_MISSING")
        return record
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    conf = json.loads(config_path.read_text(encoding="utf-8"))
    if conf.get("reference_sha256") != TRAIN_REFERENCE_SHA256:
        raise RuntimeError("Reference hash unexpectedly changed during screening")
    # These are SAMPLE-level empirical thresholds; NOT sufficient to
    # prove mathematical closed-loop stability or all-box feasibility.
    completed = bool(data.get("reached_end", False))
    success = float(data.get("solver_success_pct", 0.))
    aborted = bool(data.get("solver_aborted", True))
    feasible = bool(data.get("tracking_feasible", False))
    healthy = completed and success >= 99.9 and not aborted and feasible
    record.update(
        status="EMPIRICAL_PASS" if healthy else "EMPIRICAL_FAIL",
        reached_end=completed, solver_success_pct=success,
        tracking_feasible=feasible, solver_aborted=aborted,
        mean_cte_post_m=data.get("mean_cte_post_m"),
        max_cte_post_m=data.get("max_cte_post_m"),
        rmse_heading_post_rad=data.get("rmse_heading_post_rad"),
        rmse_vx_post_mps=data.get("rmse_vx_post_mps"),
        completion_pct=data.get("completion_pct"),
        solve_p95_ms=data.get("solve_p95_ms"),
    )
    return record


def _write_results(path, cases):
    with path.open("w", newline="", encoding="utf-8") as output:
        keys = [
            "label", "ey_m", "heading_deg", "status",
            "reached_end", "solver_success_pct",
            "tracking_feasible", "solver_aborted",
            "mean_cte_post_m", "max_cte_post_m",
            "rmse_heading_post_rad", "rmse_vx_post_mps",
            "completion_pct", "solve_p95_ms", "returncode",
            "case_dir", "error", "q", "r", "p",
        ]
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        for case in cases:
            row = {k: case.get(k, "") for k in keys}
            for key in ("q", "r", "p"):
                row[key] = json.dumps(row[key])
            writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, required=True,
                        help="Exploratory relative endpoint margin, e.g. 0.20")
    parser.add_argument("--profile", choices=("quick", "full"),
                        default="quick")
    parser.add_argument("--seconds", type=float, default=150.)
    args = parser.parse_args(argv)
    if not np.isfinite(args.seconds) or args.seconds <= 0.:
        parser.error("--seconds must be finite and positive")
    try:
        ref, digest = verify_train_reference(args.reference)
        box = screening_box(args.margin)
        proposals = representative_weights(args.margin, args.profile)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    root = args.output.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("Output must be a new/empty folder; existing evidence protected")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "EXPLORATORY_ONLY_NOT_CERTIFIED",
        "train_reference": str(ref), "train_sha256": digest,
        "profile": args.profile, "margin_fraction": args.margin,
        "simulation_seconds_limit": args.seconds,
        "learning_initial": [v.tolist() for v in LEARNING_INITIAL.arrays()],
        "strong_fixed_baseline": [v.tolist() for v in GOOD_NMPC.arrays()],
        "screen_candidate_lower": [v.tolist() for v in box[0]],
        "screen_candidate_upper": [v.tolist() for v in box[1]],
        "initial_conditions": initial_conditions(args.profile),
        "note": (
            "Finite closed-loop samples only. Passing does not establish "
            "stability or recursive feasibility across the full box; "
            "terminal invariant set and local controller are not proved."
        ),
    }
    (root / "manifest.json").write_text(json.dumps(manifest,indent=2),
                                         encoding="utf-8")
    cases = []
    for name, qrp in proposals.items():
        for ey, deg in initial_conditions(args.profile):
            result = screen_one(ref, root, name, qrp, ey, deg, args.seconds)
            cases.append(result)
            _write_results(root / "screening.csv", cases)
            print(json.dumps({
                "case": len(cases), "name": name,
                "ey": ey, "heading_deg": deg,
                "status": result["status"],
                "mean_cte_post_m": result.get("mean_cte_post_m"),
            }), flush=True)

    all_pass = all(c["status"] == "EMPIRICAL_PASS" for c in cases)
    manifest.update(total_samples=len(cases),
                    pass_samples=sum(c["status"]=="EMPIRICAL_PASS" for c in cases),
                    empirical_all_pass=all_pass,
                    conclusion=(
                        "Finite sampled conditions passed, but parameter box "
                        "NOT stability-certified; theoretical checks still pending."
                        if all_pass else
                        "One or more sampled conditions failed; box not acceptable "
                        "for training without revision."
                    ))
    (root / "manifest.json").write_text(json.dumps(manifest,indent=2),
                                         encoding="utf-8")
    print(json.dumps({
        "empirical_all_pass": all_pass, "samples": len(cases),
        "results": str(root / "screening.csv"),
        "certified_stable": False,
    }), flush=True)
    return 0 if all_pass else 1


if __name__=="__main__":
    raise SystemExit(main())
