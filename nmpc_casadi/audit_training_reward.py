"""TRAIN-only paper Eq. (4) reward and frozen initial-Critic TD audit.

Old RL reward Qe/Re is a CANDIDATE: both matrices stay constant across all
comparators/episodes and NEVER track Actor's changing Q/R. Compute from
per-step fixed-NMPC CSV, not aggregate CTE summaries.

Usage:
 python audit_training_reward.py --screen-root results_screen/full_20pct \
     --output results_reward_audit/audit01 --gamma 0.99

The CLI gamma is an EXPLICIT user-supplied diagnostic choice, not a
source-paper numerical constant. No Actor or Critic parameters are updated.
"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from direct_p_critic import error_from_state
from rl_paper_core import paper_reward, td_error
from training_initialization import LEARNING_INITIAL, TRAIN_REFERENCE_SHA256

Q_EVAL_CANDIDATE = np.array([20., 20., 100., 20., 600., 600.])
R_EVAL_CANDIDATE = np.array([50., 100.])
STATE = ("vx", "vy", "psi", "r", "X", "Y")


def reward_diagnostics(df, *, gamma, p_initial=None, q_eval=None, r_eval=None):
    """Diagnostic current-state e and true applied input difference Δu."""
    if not len(df):
        raise ValueError("Empty trajectory log")
    qe=Q_EVAL_CANDIDATE if q_eval is None else np.asarray(q_eval,float)
    re=R_EVAL_CANDIDATE if r_eval is None else np.asarray(r_eval,float)
    p=LEARNING_INITIAL.arrays()[2] if p_initial is None else np.asarray(p_initial,float)
    if p.shape != (6,) or not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("P must be six positive finite values")
    needed = [f"e_{n}" for n in STATE] + ["du_a","du_delta"]
    missing=[n for n in needed if n not in df]
    if missing:
        raise ValueError("Missing full-state error or actual control increments: "+",".join(missing))
    e=df[[f"e_{n}" for n in STATE]].to_numpy(dtype=float,copy=True)
    e[:,2]=np.arctan2(np.sin(e[:,2]),np.cos(e[:,2]))
    du=df[["du_a","du_delta"]].to_numpy(float)
    reward = np.asarray(paper_reward(e,du,qe,re),float)
    values=-(e**2)@p
    if len(e)>1:
        d=np.asarray(td_error(reward[:-1],values[:-1],values[1:],gamma))
    else:
        d=np.empty(0,float)
    if not np.all(np.isfinite(reward)) or not np.all(np.isfinite(d)):
        raise ValueError("Non-finite reward or TD")
    result={
        "steps":int(len(e)),
        "mean_reward":float(np.mean(reward)),
        "min_reward":float(np.min(reward)),
        "max_reward":float(np.max(reward)),
        "total_reward":float(np.sum(reward)),
        "reward_rms":float(np.sqrt(np.mean(reward**2))),
        "mean_tracking_penalty":float(np.mean(np.sum(e**2*qe,axis=1))),
        "mean_control_increment_penalty":float(np.mean(np.sum(du**2*re,axis=1))),
        "critic_initial_value_mean":float(np.mean(values)),
        "td_transitions_nonterminal":int(len(d)),
        "td_mean":float(np.mean(d)) if len(d) else None,
        "td_rms":float(np.sqrt(np.mean(d**2))) if len(d) else None,
        "td_abs_max":float(np.max(np.abs(d))) if len(d) else None,
    }
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screen-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--gamma",type=float,required=True)
    args=ap.parse_args(argv)
    from rl_paper_core import td_error
    td_error(0.,0.,0.,args.gamma)
    root=args.screen_root.expanduser().resolve(strict=True)
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("train_sha256") != TRAIN_REFERENCE_SHA256:
        ap.error("Screening did not use the pinned TRAIN source")
    out=args.output.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        ap.error("Use an empty output directory, prior diagnostics protected")
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    cases=pd.read_csv(root/"screening.csv")
    for _,case in cases.iterrows():
        if case["status"] != "EMPIRICAL_PASS":
            raise RuntimeError("Refuse reward comparison with failed screening case")
        dir=Path(case["case_dir"])
        if not dir.is_absolute():
            dir=(root/dir).resolve()
        conf=json.loads((dir/"config.json").read_text(encoding="utf-8"))
        if conf.get("reference_sha256")!=TRAIN_REFERENCE_SHA256:
            raise RuntimeError("Unexpected TRAIN reference in "+str(dir))
        data=pd.read_csv(dir/"nmpc_fixed.csv")
        row=reward_diagnostics(data,gamma=args.gamma)
        row.update(case=str(case["label"]),ey_m=float(case["ey_m"]),
                   heading_deg=float(case["heading_deg"]))
        rows.append(row)
    table=pd.DataFrame(rows)
    table.to_csv(out/"reward_audit.csv",index=False)
    summary={
        "type":"TRAIN_ONLY_FIXED_REWARD_CANDIDATE_NO_LEARNING",
        "train_sha256":TRAIN_REFERENCE_SHA256,
        "Qe":Q_EVAL_CANDIDATE.tolist(),
        "Re":R_EVAL_CANDIDATE.tolist(),
        "gamma_diagnostic":args.gamma,
        "P_initial":LEARNING_INITIAL.arrays()[2].tolist(),
        "cases":len(rows),
        "warning":"TD computed only on logged consecutive pre-step state pairs; "
                  "last transition omitted; no terminal bootstrap correction. "
                  "Audit does not prove stable Actor-Critic learning.",
        "mean_reward_range": [float(table.mean_reward.min()),float(table.mean_reward.max())],
        "td_rms_range":[float(table.td_rms.min()),float(table.td_rms.max())],
        "reward_audit_file":str(out/"reward_audit.csv"),
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
