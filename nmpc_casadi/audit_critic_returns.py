"""TRAIN-ONLY discounted return / direct-six-P critic representability audit.

This is an OFFLINE diagnostic, not TD learning or policy evaluation of RL.
Loads all 27 previously screened *fixed-weight* TRAIN trajectories and
compares the observed finite-episode cost-to-go G_t against
    -V_P(s_t) = sum_i p_i * e_i(t)^2.

Three comparisons:
  1. User-approved initial P.
  2. Best P within the screened box (coordinate-wise convex least squares).
  3. Unrestricted SIGNED least-squares P, a NON-ADMISSIBLE mathematical
     lower bound on error. Negative entries must never be used in NMPC.

Uses the SAME fixed paper Eq. (4) reward as the previous reward audit.
No reward, P, gamma, learning-rate or Actor parameter is changed.
Gamma choices must be explicit. TEST is neither read nor accepted.

Logged fixed-NMPC CSVs contain pre-step errors and applied du but not
the last post-step state. With reached_end=true, the last LOGGED reward
is included and the unobserved tail is treated as zero. Consequently,
the targets are finite recorded-episode returns, not a certified exact
infinite-horizon V(s). This matters especially near the final step.

PowerShell from nmpc_casadi:
 python audit_critic_returns.py `
   --screen-root .\results_screen\full_20pct `
   --output .\results_critic_audit\gamma_sweep_v1 `
   --gammas 0.90 0.95 0.99
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_training_reward import Q_EVAL_CANDIDATE, R_EVAL_CANDIDATE, STATE
from rl_paper_core import paper_reward, td_error
from screen_training_weights import verify_train_reference
from training_initialization import LEARNING_INITIAL, TRAIN_REFERENCE_SHA256, screening_box

EXPECTED_CASES = 27
EXPECTED_CONDITIONS = ((0.05, 0.), (-0.05, 0.), (0.05, 5.))


def observed_discounted_costs(reward, gamma):
    """For finite recorded episode r[0:T], C[t]=-sum_{j=t}^{T-1} gamma^(j-t) r[j].

    Zero beyond last recorded transition is a finite-episode boundary
    convention, NOT an assumption that discounted infinite-horizon cost
    to go is zero for arbitrarily truncated runs.
    """
    r = np.asarray(reward, dtype=float)
    td_error(0., 0., 0., gamma)  # validate gamma from paper core
    if r.ndim != 1 or r.size == 0 or not np.all(np.isfinite(r)):
        raise ValueError("reward must be a nonempty finite vector")
    out = np.empty_like(r)
    acc = 0.
    for i in range(len(r) - 1, -1, -1):
        acc = -r[i] + gamma * acc
        out[i] = acc
    return out


def error_features_and_reward(frame):
    """Quadratic six-state columns and fixed Eq. (4) per-sample reward.

    Must have precisely the same physical state errors/du as the live
    Actor-Critic reward. Heading wraps exactly as in NMPC.
    """
    fields = [f"e_{s}" for s in STATE] + ["du_a", "du_delta"]
    missing = [name for name in fields if name not in frame.columns]
    if missing or frame.empty:
        raise ValueError(f"Empty/missing full-state NMPC log columns: {missing}")
    e = frame[[f"e_{s}" for s in STATE]].to_numpy(dtype=float, copy=True)
    e[:, 2] = np.arctan2(np.sin(e[:, 2]), np.cos(e[:, 2]))
    du = frame[["du_a", "du_delta"]].to_numpy(dtype=float)
    reward = np.asarray(
        paper_reward(e, du, Q_EVAL_CANDIDATE, R_EVAL_CANDIDATE),
        dtype=float,
    )
    if not np.all(np.isfinite(e)) or not np.all(np.isfinite(reward)):
        raise ValueError("Nonfinite state errors or reward in log")
    return e**2, reward


def fitted_box_least_squares(features, target, lower, upper, initial,
                             *, max_cycles=30000, tolerance=1e-11):
    """Convex coordinate descent in column-normalized 6D coordinates.

    Each step exactly minimizes squared error over one coordinate then
    projects to the specified box. Returns fit + convergence flag; no
    optimizer dependency beyond NumPy. This is batch RETROSPECTIVE
    diagnosis, never the paper TD(0) Critic update.
    """
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    low, high, start = (np.asarray(v, dtype=float) for v in
                        (lower, upper, initial))
    if (x.ndim != 2 or x.shape[1] != 6 or y.shape != (len(x),) or
        any(z.shape != (6,) for z in (low, high, start)) or
        not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or
        np.any(low <= 0) or np.any(high <= low) or
        np.any(start < low) or np.any(start > high)):
        raise ValueError("Need positive finite P box and finite (n,6) data")
    scales = np.linalg.norm(x, axis=0)
    inactive = scales < 1e-14
    scales[inactive] = 1.
    z = x / scales
    g = z.T @ z
    h = z.T @ y
    theta = start * scales
    lo = low * scales
    hi = high * scales
    converged = False
    for cycle in range(1, max_cycles + 1):
        largest_step = 0.
        for i in range(6):
            if inactive[i]:
                continue
            update = (h[i] - float(g[i, :] @ theta)) / g[i, i]
            val = float(np.clip(theta[i] + update, lo[i], hi[i]))
            largest_step = max(largest_step, abs(val - theta[i]))
            theta[i] = val
        if largest_step <= tolerance * max(1., float(np.linalg.norm(theta))):
            converged = True
            break
    return theta / scales, {"converged": converged, "cycles": cycle,
                            "inactive_columns": [int(i) for i in np.flatnonzero(inactive)]}


def signed_unrestricted_least_squares(features, target):
    """Only a mathematical lower bound; may violate positive P."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or x.shape[1] != 6 or y.shape != (len(x),):
        raise ValueError("features (n,6), target (n,) required")
    scales = np.linalg.norm(x, axis=0)
    inactive = scales < 1e-14
    scales[inactive] = 1.
    pscaled, _, rank, sv = np.linalg.lstsq(x / scales, y, rcond=None)
    return pscaled / scales, int(rank)


def fit_metrics(features, target, p):
    x = np.asarray(features, float)
    y = np.asarray(target, float)
    pp = np.asarray(p, float)
    pred = x @ pp
    diff = pred - y
    denom = float(np.sum((y - np.mean(y))**2))
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "nrmse_over_rms_target": float(np.sqrt(np.mean(diff**2)) /
                                         max(float(np.sqrt(np.mean(y**2))), 1e-15)),
        "r2": float(1. - np.dot(diff, diff) / denom) if denom > 1e-20 else None,
        "mean_predicted_cost": float(np.mean(pred)),
        "mean_observed_cost": float(np.mean(y)),
    }


def _resolve_case_dir(root, raw):
    """Case dir can be absolute (the legacy screening writes absolute paths)."""
    d = Path(str(raw))
    return d if d.is_absolute() else (root / d)


def load_full_train_screening(root):
    """Fail closed unless all 27 locally logged cases match pinned TRAIN.

    Verifies manifest, each case's config and summary; never uses TEST.
    """
    root = Path(root).expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "EXPLORATORY_ONLY_NOT_CERTIFIED" or
        manifest.get("profile") != "full" or
        manifest.get("train_sha256") != TRAIN_REFERENCE_SHA256 or
        manifest.get("total_samples") != EXPECTED_CASES or
        manifest.get("pass_samples") != EXPECTED_CASES or
        manifest.get("empirical_all_pass") is not True):
        raise ValueError("Requires 27/27 full screening on pinned TRAIN")
    candidate_lower, candidate_upper = screening_box(.20)
    for supplied, expected in zip(
        manifest["screen_candidate_lower"], candidate_lower
    ):
        np.testing.assert_allclose(supplied, expected, rtol=1e-10, atol=1e-10)
    for supplied, expected in zip(
        manifest["screen_candidate_upper"], candidate_upper
    ):
        np.testing.assert_allclose(supplied, expected, rtol=1e-10, atol=1e-10)
    # Importantly, NEVER use a held-out/test reference as input or as a fit target.
    train_path = Path(manifest["train_reference"])
    reference, sha = verify_train_reference(train_path)
    if sha != TRAIN_REFERENCE_SHA256:
        raise ValueError("TRAIN SHA256 changed")

    cases = pd.read_csv(root / "screening.csv")
    if len(cases) != EXPECTED_CASES or set(cases["status"]) != {"EMPIRICAL_PASS"}:
        raise ValueError("Full screen must have exactly 27 PASS rows")
    seen = set()
    records = []
    for row in cases.itertuples(index=False):
        cond = (round(float(row.ey_m), 8), round(float(row.heading_deg), 8))
        key = (str(row.label), cond[0], cond[1])
        if key in seen:
            raise ValueError(f"Duplicate screening case {key}")
        seen.add(key)
        if cond not in EXPECTED_CONDITIONS:
            raise ValueError(f"Unknown condition {cond}")
        directory = _resolve_case_dir(root, row.case_dir)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if (config.get("reference_sha256") != TRAIN_REFERENCE_SHA256 or
            not bool(summary.get("reached_end")) or
            bool(summary.get("solver_aborted")) or
            not bool(summary.get("tracking_feasible")) or
            float(summary.get("solver_success_pct", 0.)) < 99.9):
            raise ValueError(f"Invalid TRAIN/closed-loop case {directory}")
        for name, csv_column in (("Q_diagonal", "q"),("R_diagonal","r"),
                                 ("Qf_diagonal","p")):
            values = np.asarray(json.loads(getattr(row, csv_column)), float)
            np.testing.assert_allclose(config[name], values, rtol=1e-9, atol=1e-9)
        frame = pd.read_csv(directory / "nmpc_fixed.csv")
        if len(frame) < 2:
            raise ValueError(f"Too few samples in {directory}")
        if "solver_success" not in frame or not frame["solver_success"].all():
            raise ValueError(f"Only fully successful fixed NMPC samples accepted: {directory}")
        x, reward = error_features_and_reward(frame)
        records.append({
            "label": str(row.label), "ey_m": cond[0], "heading_deg": cond[1],
            "features": x, "reward": reward, "n_steps": len(frame),
        })
    labels = {r["label"] for r in records}
    if len(labels) != 9 or any(
        sum(r["label"] == lbl for r in records) != 3 for lbl in labels
    ):
        raise ValueError("Require nine unique weight configurations x three offsets")
    return records


def evaluate_discount(gamma, records, lower, upper, initial):
    """Return in-sample fits + leave-one-initial-condition-out diagnostics."""
    features = np.concatenate([r["features"] for r in records])
    targets_by_case = [observed_discounted_costs(r["reward"], gamma)
                       for r in records]
    targets = np.concatenate(targets_by_case)
    pbox, conv = fitted_box_least_squares(features, targets, lower, upper, initial)
    signed, rank = signed_unrestricted_least_squares(features, targets)
    initial_report = fit_metrics(features, targets, initial)
    box_report = fit_metrics(features, targets, pbox)
    signed_report = fit_metrics(features, targets, signed)
    results = []
    for rec, costs in zip(records, targets_by_case):
        results.append({
            "label": rec["label"], "ey_m": rec["ey_m"],
            "heading_deg": rec["heading_deg"], "gamma": gamma,
            "steps": rec["n_steps"],
            "start_observed_cost": float(costs[0]),
            "end_logged_cost": float(costs[-1]),
            "initial_rmse": fit_metrics(rec["features"], costs, initial)["rmse"],
            "box_fit_rmse": fit_metrics(rec["features"], costs, pbox)["rmse"],
            "initial_cost_predicted": float(rec["features"][0] @ initial),
            "box_cost_predicted": float(rec["features"][0] @ pbox),
        })
    held_out = []
    for cond in EXPECTED_CONDITIONS:
        in_fold = [i for i, r in enumerate(records)
                   if (r["ey_m"], r["heading_deg"]) != cond]
        out_fold = [i for i, r in enumerate(records)
                    if (r["ey_m"], r["heading_deg"]) == cond]
        ftrain = np.concatenate([records[i]["features"] for i in in_fold])
        ytrain = np.concatenate([targets_by_case[i] for i in in_fold])
        fvalid = np.concatenate([records[i]["features"] for i in out_fold])
        yvalid = np.concatenate([targets_by_case[i] for i in out_fold])
        fold_p, convergence = fitted_box_least_squares(
            ftrain, ytrain, lower, upper, initial
        )
        held_out.append({
            "heldout_ey_m": cond[0], "heldout_heading_deg": cond[1],
            "fit_converged": convergence["converged"],
            "trained_p": [float(v) for v in fold_p],
            "initial": fit_metrics(fvalid, yvalid, initial),
            "train_fit": fit_metrics(ftrain, ytrain, fold_p),
            "heldout_fit": fit_metrics(fvalid, yvalid, fold_p),
        })
    return {
        "gamma": float(gamma),
        "sample_count": len(targets),
        "n_cases": len(records),
        "fit_rank_unrestricted": rank,
        "p_initial": [float(v) for v in initial],
        "p_screen_box_fit": [float(v) for v in pbox],
        "p_signed_unrestricted_diagnostic_only": [float(v) for v in signed],
        "box_optimization": conv,
        "initial_metrics": initial_report,
        "box_fit_metrics_in_sample": box_report,
        "signed_unrestricted_metrics_in_sample_lower_bound": signed_report,
        "p_box_at_lower": [bool(v) for v in
                            np.isclose(pbox, lower, rtol=1e-7, atol=1e-9)],
        "p_box_at_upper": [bool(v) for v in
                            np.isclose(pbox, upper, rtol=1e-7, atol=1e-9)],
        "heldout_within_train_by_initial_condition": held_out,
        "cases": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gammas", nargs="+", type=float, required=True,
                        help="Explicit gamma diagnostic choices; no gamma is auto-approved")
    args = parser.parse_args(argv)
    if not args.gammas or len(set(args.gammas)) != len(args.gammas):
        parser.error("Provide unique gamma values")
    for gamma in args.gammas:
        td_error(0., 0., 0., gamma)
    try:
        records = load_full_train_screening(args.screen_root)
    except (ValueError, OSError, KeyError, AssertionError) as exc:
        parser.error(str(exc))
    out = args.output.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory must be new/empty (prior results protected)")
    out.mkdir(parents=True, exist_ok=True)
    lower, upper = screening_box(.20)
    plow, phigh = lower[2], upper[2]
    p0 = LEARNING_INITIAL.arrays()[2]
    reports = []
    for gamma in args.gammas:
        report = evaluate_discount(gamma, records, plow, phigh, p0)
        reports.append(report)
        print(json.dumps({
            "gamma": gamma, "case_count": report["n_cases"],
            "sample_count": report["sample_count"],
            "initial_rmse": report["initial_metrics"]["rmse"],
            "box_fit_rmse": report["box_fit_metrics_in_sample"]["rmse"],
            "signed_unrestricted_rmse_lower_bound":
                report["signed_unrestricted_metrics_in_sample_lower_bound"]["rmse"],
            "box_optimizer_converged": report["box_optimization"]["converged"],
            "p_at_upper": report["p_box_at_upper"],
        }), flush=True)
    pd.DataFrame([
        item for r in reports for item in r["cases"]
    ]).to_csv(out / "return_cases.csv", index=False)
    pd.DataFrame([
        {"gamma": r["gamma"], "initial_rmse": r["initial_metrics"]["rmse"],
         "screen_box_fit_rmse": r["box_fit_metrics_in_sample"]["rmse"],
         "unrestricted_signed_rmse": r[
             "signed_unrestricted_metrics_in_sample_lower_bound"]["rmse"],
         "screen_box_nrmse": r[
             "box_fit_metrics_in_sample"]["nrmse_over_rms_target"],
         "box_fit_converged": r["box_optimization"]["converged"],
         "unrestricted_rank": r["fit_rank_unrestricted"]}
        for r in reports
    ]).to_csv(out / "return_gamma_comparison.csv", index=False)
    summary = {
        "type": "TRAIN_ONLY_FINITE_EPISODE_DIRECT_P_REPRESENTABILITY_AUDIT",
        "train_sha256": TRAIN_REFERENCE_SHA256,
        "fixed_Qe": Q_EVAL_CANDIDATE.tolist(),
        "fixed_Re": R_EVAL_CANDIDATE.tolist(),
        "gamma_diagnostics": list(args.gammas),
        "screening_cases": EXPECTED_CASES,
        "reference_role": "TRAIN_ONLY",
        "no_learning": True,
        "box_is_certified_stable": False,
        "fit_is_batch_diagnostic_not_TD0_training": True,
        "signed_unrestricted_fit_is_not_positive_admissible": True,
        "return_boundary": "Finite recorded episode: last logged reward included; future beyond CSV=0. Cannot establish exact infinite-horizon value or final post-step bootstrap.",
        "condition_holdout": "Leave-one-initial-condition-out WITHIN TRAIN, not a held-out TEST set",
        "reports": reports,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"saved_to": str(out), "gammas": args.gammas,
                      "cases": EXPECTED_CASES, "training_performed": False}),
          flush=True)


if __name__ == "__main__":
    main()
