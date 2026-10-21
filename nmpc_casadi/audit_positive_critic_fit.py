"""Second-stage TRAIN-only diagnostic: nonnegative P without upper bound.

WHY: the original 27-case fit hits all SIX P upper limits for every
tested gamma. A signed unconstrained least-squares fit can be good but
uses negative coefficients (inadmissible for terminal Qf). We must
measure the intermediate, physical-sign-constrained approximation.

The positive *unbounded above* NNLS fit is STILL NOT NMPC-admissible:
no terminal-set, recursive-feasibility or solver certification is
implied. Never automatically change critic.p or its bounds using this.

Furthermore, all 9 screened fixed NMPC controllers differ. Each
fixed-policy return is policy-dependent. Compare pooled NNLS to nine
separate policy-specific NNLS fits before diagnosing general Critic
function-class limitations.

For 6-dimensional P, an exhaustive 2^6 active-subset NNLS solver is
possible with NumPy alone. A single thin QR factorization of the
column-scaled observation matrix permits numerically stable 6x6
subset least squares. No SciPy, batch learning, or change to training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_critic_returns import (
    EXPECTED_CASES, EXPECTED_CONDITIONS, fit_metrics,
    load_full_train_screening, observed_discounted_costs,
    signed_unrestricted_least_squares,
)
from audit_training_reward import Q_EVAL_CANDIDATE, R_EVAL_CANDIDATE
from training_initialization import TRAIN_REFERENCE_SHA256, screening_box


def positive_no_upper_least_squares(features, target):
    """Exact active-set enumeration for NNLS min ||X p-y||², p>=0, six p.

    Rescale columns then thin QR once, avoiding normal equations and
    squaring condition numbers. Enumerate all 64 support subsets:
    for each support S solve R[:,S] theta_S ~= Q.T y, retain only
    theta>=0, choose smallest observed squared residual. Zero active
    set is included. The KKT residual is checked and reported; if it
    is large the caller must treat the result as numerically suspect.

    This fit is a RETROSPECTIVE mathematical diagnostic; nonnegative
    P alone is NOT proof that it is an admissible NMPC terminal weight.
    """
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    if (x.ndim != 2 or x.shape[1] != 6 or y.shape != (len(x),)
            or len(x) < 6 or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y))):
        raise ValueError("Expected finite features (n>=6, 6) and target (n,)")
    norms = np.linalg.norm(x, axis=0)
    small = norms < 1e-15
    safe_norms = np.where(small, 1., norms)
    z = x / safe_norms
    q, r = np.linalg.qr(z, mode="reduced")
    projection = q.T @ y
    base = float(np.linalg.norm(y)**2 - np.linalg.norm(projection)**2)
    best = np.zeros(6, dtype=float)
    best_sq = max(0., base) + float(np.dot(projection, projection))
    best_support = 0
    # Exhaustive support enumeration; optimal NNLS has an active support.
    for mask in range(1, 1 << 6):
        idx = np.flatnonzero([(mask >> i) & 1 for i in range(6)])
        coeff, *_ = np.linalg.lstsq(r[:, idx], projection, rcond=None)
        # A negative coefficient is infeasible for this active support.
        if np.any(coeff < -1e-10 * max(1., np.max(np.abs(coeff)))):
            continue
        coeff = np.maximum(coeff, 0.)
        v = np.zeros(6)
        v[idx] = coeff
        delta = r @ v - projection
        sq = max(0., base) + float(delta @ delta)
        if sq < best_sq:
            best_sq = sq
            best = v.copy()
            best_support = mask

    scaled_gradient = r.T @ (r @ best - projection)
    stationary_tol = 1e-7 * max(1., float(np.linalg.norm(r.T @ projection)))
    kkt_inactive = float(np.max(np.maximum(-scaled_gradient[best <= 1e-10], 0.),
                                  initial=0.))
    kkt_active = float(np.max(np.abs(scaled_gradient[best > 1e-10]),
                              initial=0.))
    status = {
        "method": "six_coordinate_exhaustive_nnls_qr",
        "support_mask": int(best_support),
        "active_coordinates": [int(i) for i in np.flatnonzero(best > 1e-10)],
        "rank": int(np.linalg.matrix_rank(r)),
        "max_kkt_violation": max(kkt_active, kkt_inactive),
        "kkt_tolerance": stationary_tol,
        "kkt_pass": bool(max(kkt_active, kkt_inactive) <= stationary_tol),
    }
    return best / safe_norms, status


def diagnose_gamma(gamma, records):
    """Pooled and policy-specific fits, all on the pinned TRAIN track."""
    costs = [observed_discounted_costs(r["reward"], gamma) for r in records]
    data = np.concatenate([r["features"] for r in records])
    target = np.concatenate(costs)
    pooled_p, pooled_status = positive_no_upper_least_squares(data, target)
    pooled_metrics = fit_metrics(data, target, pooled_p)
    p_low, p_high = (v[2] for v in screening_box(.20))
    labels = sorted({r["label"] for r in records})
    per_policy = []
    for label in labels:
        ids = [i for i, rec in enumerate(records) if rec["label"] == label]
        if len(ids) != 3:
            raise ValueError(f"Expected three TRAIN initial conditions for {label}")
        fx = np.concatenate([records[i]["features"] for i in ids])
        fy = np.concatenate([costs[i] for i in ids])
        best_p, status = positive_no_upper_least_squares(fx, fy)
        signed_p, rank = signed_unrestricted_least_squares(fx, fy)
        per_policy.append({
            "label": label, "gamma": gamma, "steps": len(fy),
            "p_nnls_nonnegative_unbounded_not_for_nmpc": best_p.tolist(),
            "nnls": fit_metrics(fx, fy, best_p),
            "pooled_positive_fit_on_this_policy": fit_metrics(fx, fy, pooled_p),
            "signed_unrestricted_fit": fit_metrics(fx, fy, signed_p),
            "signed_fit_has_negative": bool(np.any(signed_p < 0.)),
            "rank": rank,
            "nnls_kkt_pass": status["kkt_pass"],
            "nnls_max_kkt_violation": status["max_kkt_violation"],
        })
    # Honest interpretation: pooled data mix different fixed policies;
    # leave-policy-out cross-generalization is not an actual adaptive policy.
    return {
        "gamma": float(gamma),
        "cases": len(records), "samples": len(target),
        "pooled_p_nonnegative_no_upper_not_nmpc_certified": pooled_p.tolist(),
        "pooled_nnls": pooled_metrics,
        "pooled_nnls_numerical_check": pooled_status,
        "pooled_p_exceeds_prior_P_upper": [bool(z) for z in
                                          pooled_p > p_high + 1e-8],
        "pooled_p_below_prior_P_lower": [bool(z) for z in
                                         pooled_p < p_low - 1e-8],
        "per_fixed_policy": per_policy,
        "mean_policy_nnls_rmse": float(np.mean([
            item["nnls"]["rmse"] for item in per_policy])),
        "mean_policy_pooled_rmse": float(np.mean([
            item["pooled_positive_fit_on_this_policy"]["rmse"]
            for item in per_policy])),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screen-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--gammas", nargs="+", type=float, required=True)
    args = ap.parse_args(argv)
    if len(set(args.gammas)) != len(args.gammas):
        ap.error("Repeated gamma values")
    if any(not np.isfinite(g) or not (0. < g <= 1.) for g in args.gammas):
        ap.error("Require 0 < gamma <= 1")
    records = load_full_train_screening(args.screen_root)
    out = args.output.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        ap.error("Refuse to overwrite nonempty diagnostic output")
    out.mkdir(parents=True, exist_ok=True)
    reports = [diagnose_gamma(g, records) for g in args.gammas]
    if not all(item["pooled_nnls_numerical_check"]["kkt_pass"] for item in reports):
        raise RuntimeError("NNLS KKT check failed: do not interpret numerical fit")
    if not all(p["nnls_kkt_pass"] for item in reports for p in item["per_fixed_policy"]):
        raise RuntimeError("Per-policy NNLS KKT check failed")
    summary = {
        "type": "TRAIN_ONLY_POSITIVE_NO_UPPER_DIRECT_P_DIAGNOSTIC",
        "train_sha256": TRAIN_REFERENCE_SHA256,
        "fixed_Qe": Q_EVAL_CANDIDATE.tolist(),
        "fixed_Re": R_EVAL_CANDIDATE.tolist(),
        "gamma_diagnostics": args.gammas,
        "screening_cases": EXPECTED_CASES,
        "no_training": True,
        "no_test_data": True,
        "not_stability_certified": True,
        "nonnegative_zero_allowed_for_diagnostic_only": True,
        "p_policy_dependence_note": "Nine different fixed NMPC policies produce different returns even at shared initial states; policy-specific fits are not a single adaptive-Actor V(s).",
        "finite_return_boundary_note": "Full recorded episode only; final unrecorded post-state not represented. Future reward beyond final recorded step is zero.",
        "reports": reports,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame([{
        "gamma": r["gamma"],
        "pooled_positive_rmse": r["pooled_nnls"]["rmse"],
        "pooled_positive_nrmse": r["pooled_nnls"]["nrmse_over_rms_target"],
        "pooled_positive_r2": r["pooled_nnls"]["r2"],
        "pooled_p_exceeds_box_upper_count":
            sum(r["pooled_p_exceeds_prior_P_upper"]),
        "mean_policy_nnls_rmse": r["mean_policy_nnls_rmse"],
        "mean_policy_pooled_rmse": r["mean_policy_pooled_rmse"],
        "pooled_nnls_kkt_pass": r["pooled_nnls_numerical_check"]["kkt_pass"],
    } for r in reports]).to_csv(out / "positive_gamma_comparison.csv", index=False)
    pd.DataFrame([{
        "gamma": r["gamma"], "label": p["label"],
        "steps": p["steps"],
        "policy_nnls_rmse": p["nnls"]["rmse"],
        "policy_nnls_nrmse": p["nnls"]["nrmse_over_rms_target"],
        "pooled_nnls_rmse_on_policy":
            p["pooled_positive_fit_on_this_policy"]["rmse"],
        "unrestricted_signed_rmse": p["signed_unrestricted_fit"]["rmse"],
        "signed_has_negative": p["signed_fit_has_negative"],
        "policy_nnls_kkt_pass": p["nnls_kkt_pass"],
    } for r in reports for p in r["per_fixed_policy"]]).to_csv(
        out / "positive_policy_comparison.csv", index=False)
    for item in reports:
        print(json.dumps({
            "gamma": item["gamma"],
            "pooled_positive_nrmse":
                item["pooled_nnls"]["nrmse_over_rms_target"],
            "positive_p_above_prior_upper":
                sum(item["pooled_p_exceeds_prior_P_upper"]),
            "per_policy_mean_nnls_rmse": item["mean_policy_nnls_rmse"],
            "pooled_mean_rmse_on_policies": item["mean_policy_pooled_rmse"],
        }), flush=True)
    print(json.dumps({"status":"DIAGNOSTIC_NO_LEARNING",
                      "output":str(out),"case_count":len(records)}))


if __name__=="__main__":
    main()
