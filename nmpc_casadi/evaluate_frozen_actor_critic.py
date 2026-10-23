"""Frozen Actor–Critic checkpoint evaluation on the EXACT pinned TRAIN track.

No Actor sampling (deterministic mean), no Actor/Critic updates, no TEST,
and no edits to independent fixed-NMPC controls. Runs BOTH:
  (1) UNTRAINED Actor mean, initial six-P
  (2) TRAINED checkpoint Actor mean, frozen saved six-P
under exactly identical initial conditions and the same live adaptive-cost
NMPC solver/plant. Reads prior full-screening fixed NMPC trajectories for
benchmarks, recalculating identical task Eq. (4) reward. Never claims that
TRAIN result demonstrates generalization to held-out TEST.

With --all-train-conditions, run all three already screened initial
conditions; otherwise only user-approved initial +0.05m, 0deg.

This is an engineering comparison, not theoretical stability certification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import baseline as b
from audit_critic_returns import observed_discounted_costs, error_features_and_reward
from audit_training_reward import Q_EVAL_CANDIDATE, R_EVAL_CANDIDATE
from cad_eval_reference import prepare_cad_evaluation_reference
from direct_p_critic import SixParameterTDCritic
from gaussian_weight_actor import GaussianWeightActor
from screen_training_weights import verify_train_reference
from train_actor_critic import read_screened_box
from train_rl_environment import TrainEnvironment
from training_initialization import LEARNING_INITIAL, TRAIN_REFERENCE_SHA256

BASELINES = ("learning_initial","strong_fixed_baseline",
             "stage_lower_terminal_upper")
CONDITIONS = ((.05,0.),(-.05,0.),(.05,5.))


def load_frozen_agents(checkpoint, low, high, *, expected_gamma=0.90):
    """Unpickle-free, shape-validated weights for both original and learned agents."""
    q0,r0,p0=LEARNING_INITIAL.arrays()
    with np.load(checkpoint, allow_pickle=False) as saved:
        if "metadata" not in saved:
            raise ValueError("Checkpoint missing training metadata")
        config=json.loads(str(saved["metadata"].item()))
        if (config.get("reference_role")!="TRAIN_ONLY" or
            config.get("train_sha256")!=TRAIN_REFERENCE_SHA256 or
            config.get("gamma")!=expected_gamma or
            config.get("Qe_fixed")!=Q_EVAL_CANDIDATE.tolist() or
            config.get("Re_fixed")!=R_EVAL_CANDIDATE.tolist() or
            config.get("initial_Q")!=q0.tolist() or
            config.get("initial_R")!=r0.tolist() or
            config.get("initial_P")!=p0.tolist()):
            raise ValueError("Checkpoint TRAIN/reward/gamma/initial weights mismatch")
        for key,l,u in zip(("Q","R","P"),low,high):
            if (not np.allclose(config[f"{key}_lower"],l,rtol=0,atol=1e-8)
                or not np.allclose(config[f"{key}_upper"],u,rtol=0,atol=1e-8)):
                raise ValueError(f"Checkpoint {key} range mismatch")
        seed=int(config["seed"])
        settings=dict(
            nominal_q=q0, nominal_r=r0,
            q_lower=low[0],q_upper=high[0],
            r_lower=low[1],r_upper=high[1],
            sigma_lower=float(config["sigma_lower"]),
            sigma_initial=float(config["sigma_initial"]),
            sigma_upper=float(config["sigma_upper"]))
        # Both policies initialized with same neural initial values.
        initial=GaussianWeightActor(np.random.default_rng(seed), **settings)
        trained=GaussianWeightActor(np.random.default_rng(seed), **settings)
        changed=False
        for name,expected in trained.params.items():
            key="actor_"+name
            if key not in saved:
                raise ValueError(f"Missing checkpoint tensor {key}")
            actual=np.asarray(saved[key],dtype=float)
            if actual.shape!=expected.shape or not np.all(np.isfinite(actual)):
                raise ValueError(f"Invalid checkpoint tensor {key}")
            changed|=bool(np.any(actual!=expected))
            trained.params[name][...] = actual
        if not changed:
            raise ValueError("Checkpoint Actor unchanged: no learning was saved")
        if "critic_physical_P" not in saved:
            raise ValueError("Checkpoint missing Critic P")
        p=np.asarray(saved["critic_physical_P"],float)
        if p.shape!=(6,) or not np.all(np.isfinite(p)) or np.any(p<low[2]) or np.any(p>high[2]):
            raise ValueError("Frozen P outside screened experiment box")
    c0=SixParameterTDCritic(p0,low[2],high[2])
    c1=SixParameterTDCritic(p,low[2],high[2])
    return (initial,c0),(trained,c1),config


def metrics_from_rows(frame,gamma,cad_length):
    if frame.empty:
        raise ValueError("No simulation samples")
    after=frame.loc[frame["s_progress_m"]>=b.EVAL_AFTER_S_M]
    if after.empty:
        after=frame
    reward=frame["reward"].to_numpy(float)
    if not np.all(np.isfinite(reward)):
        raise ValueError("Nonfinite frozen evaluation reward")
    return {
        "steps":len(frame),"sim_time_s":float(len(frame)*b.Ts),
        "gamma":gamma,
        "discounted_cost_from_start":float(observed_discounted_costs(reward,gamma)[0]),
        "total_reward":float(np.sum(reward)),
        "mean_reward":float(np.mean(reward)),
        "mean_cte_post_m":float(after["cross_track_m"].mean()),
        "max_cte_post_m":float(after["cross_track_m"].max()),
        "rmse_heading_post_rad":float(np.sqrt(np.mean(after["e_psi"]**2))),
        "rmse_vx_post_mps":float(np.sqrt(np.mean(after["e_vx"]**2))),
        "solver_success_pct":float(100.0*frame["solver_success"].mean()),
        "completion_pct":float(100.0*frame["s_progress_m"].iloc[-1]/cad_length),
    }


def frozen_rollout(env,actor,critic,gamma,case_label):
    """Freeze both parameters, deterministic Gaussian mean, true applied-control reward."""
    before={key:value.copy() for key,value in actor.params.items()}
    frozen_p=critic.p.copy()
    obs=env.reset()
    rows=[]
    for _ in range(env.max_steps):
        import time
        loop_start=time.perf_counter()
        initial_progress=float(obs.reference["projection"]["s"])
        cte=float(obs.reference["projection"]["distance"])
        heading=float(obs.error[2])
        vx_err=float(obs.error[0])
        proposal=actor.propose(obs.features,deterministic=True)
        next_obs,reward,terminated,truncated,diag=env.step(
            proposal.q_diagonal,proposal.r_diagonal,critic,obs)
        row={
            "controller":case_label,
            "t_s":float((len(rows))*b.Ts),
            "X_m":float(obs.physical[4]),
            "Y_m":float(obs.physical[5]),
            "X_ref_m":float(obs.reference["xref"][0][4]),
            "Y_ref_m":float(obs.reference["xref"][0][5]),
            "s_progress_m":initial_progress,
            "cross_track_m":cte,
            "e_psi":heading,"e_vx":vx_err,
            "reward":reward,
            **diag,
            "terminated":terminated,"truncated":truncated,
        }
        for name,values in (("q",proposal.q_diagonal),("r",proposal.r_diagonal),
                             ("p",critic.p),("sigma",proposal.sigma)):
            for k,v in enumerate(values):
                row[f"{name}_{k}"]=float(v)
        rows.append(row)
        row["loop_ms"]=1000*(time.perf_counter()-loop_start)
        obs=next_obs
        if terminated or truncated:
            break
    for key,unchanged in before.items():
        if not np.array_equal(actor.params[key],unchanged):
            raise RuntimeError("Frozen inference unexpectedly changed Actor parameters")
    if not np.array_equal(critic.p,frozen_p):
        raise RuntimeError("Frozen inference unexpectedly changed Critic parameters")
    log=pd.DataFrame(rows)
    summary=metrics_from_rows(log,gamma,b.CAD_S_END)
    summary.update(
        controller=case_label,
        reached_end=bool(log["reached_end"].iloc[-1]),
        failure_terminal=bool(log["failure_terminal"].iloc[-1]),
        terminated=bool(log["terminated"].iloc[-1]),
        truncated=bool(log["truncated"].iloc[-1]),
        gamma=float(gamma),
        )
    if "solve_ms" in log:
        summary["solve_p95_ms"]=float(np.percentile(log["solve_ms"],95))
    return summary,log


def read_fixed_comparators(screen_root,ey,heading,gamma,cad_length):
    """Use existing full-screen fixed-NMPC logs with SAME task reward."""
    root=Path(screen_root).expanduser().resolve(strict=True)
    doc=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if (doc.get("train_sha256")!=TRAIN_REFERENCE_SHA256 or
        doc.get("empirical_all_pass") is not True or
        doc.get("profile")!="full"):
        raise ValueError("Require complete pinned TRAIN screening")
    cases=pd.read_csv(root/"screening.csv")
    reports=[]
    for label in BASELINES:
        matching=cases.loc[
            (cases["label"]==label) &
            np.isclose(cases["ey_m"].to_numpy(float),ey,atol=1e-9) &
            np.isclose(cases["heading_deg"].to_numpy(float),heading,atol=1e-9)]
        if len(matching)!=1 or matching.iloc[0]["status"]!="EMPIRICAL_PASS":
            raise ValueError(f"Missing one fixed baseline {label}, offset {ey},{heading}")
        name=Path(str(matching.iloc[0]["case_dir"]))
        directory=name if name.is_absolute() else (root/name)
        config=json.loads((directory/"config.json").read_text(encoding="utf-8"))
        if config.get("reference_sha256")!=TRAIN_REFERENCE_SHA256:
            raise ValueError("Fixed NMPC comparator TRAIN hash mismatch")
        data=pd.read_csv(directory/"nmpc_fixed.csv")
        features,reward=error_features_and_reward(data)
        required=["s_progress_m","cross_track_m","e_psi","e_vx","solver_success"]
        if not all(k in data for k in required):
            raise ValueError("Missing fixed NMPC comparator columns")
        data["reward"]=reward
        report=metrics_from_rows(data,gamma,cad_length)
        report.update(
            controller=label,
            reached_end=bool(matching.iloc[0]["reached_end"]),
            failure_terminal=bool(matching.iloc[0]["solver_aborted"]),
            terminated=bool(matching.iloc[0]["reached_end"]),
            truncated=False,
            source="previous_fixed_weight_TRAIN_screening_not_re_run",
        )
        reports.append(report)
    return reports


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference",type=Path,required=True)
    ap.add_argument("--screen-root",type=Path,required=True)
    ap.add_argument("--screen-manifest",type=Path,required=True)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--max-steps",type=int,required=True)
    ap.add_argument("--all-train-conditions",action="store_true")
    args=ap.parse_args(argv)
    if args.max_steps<1:
        ap.error("--max-steps must be positive")
    ref,digest=verify_train_reference(args.reference)
    low,high,screen=read_screened_box(args.screen_manifest)
    (initial,c0),(trained,c1),config=load_frozen_agents(
        args.checkpoint,low,high,expected_gamma=.90)
    gamma=float(config["gamma"])
    root=args.output.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        ap.error("Output exists and is nonempty; previous evaluation protected")
    root.mkdir(parents=True,exist_ok=True)
    details,staged=prepare_cad_evaluation_reference(
        ref,root,b.TAIL_LENGTH_M,
        min_speed_mps=float(b.cfg["vx_min"]),
        max_speed_mps=min(float(b.TEST_SPEED_MPS),float(b.cfg["vx_max"])),
        wheelbase_m=float(b.WHEELBASE_M),
        steering_limit_rad=min(abs(float(b.cfg["u_min"][1])),
                               abs(float(b.cfg["u_max"][1]))))
    if staged["reference_sha256"]!=digest:
        raise RuntimeError("Staged TRAIN changed")
    b.REF_INFO=details
    b.TAIL_FILE=Path(staged["tail_file"])
    b.CAD_S_END=float(details["cad_s_end"])
    q0,r0,p0=LEARNING_INITIAL.arrays()
    conditions=CONDITIONS if args.all_train_conditions else (CONDITIONS[0],)
    summaries=[]
    for ey,deg in conditions:
        env=TrainEnvironment(
            train_length_m=b.CAD_S_END,q_initial=q0,r_initial=r0,p_initial=p0,
            task_qe=Q_EVAL_CANDIDATE,task_re=R_EVAL_CANDIDATE,
            max_steps=args.max_steps,ey=ey,heading_deg=deg)
        tag=f"ey{ey:+.3f}_hdg{deg:+.1f}"
        for label,(actor,critic) in (
            ("untrained_deterministic",(initial,c0)),
            ("trained_deterministic",(trained,c1))
        ):
            record,log=frozen_rollout(env,actor,critic,gamma,label)
            log.to_csv(root/f"{label}_{tag}.csv",index=False)
            record.update(ey_m=ey,heading_deg=deg,source="frozen_NMPC_rollout")
            summaries.append(record)
            print(json.dumps(record),flush=True)
        for record in read_fixed_comparators(
            args.screen_root,ey,deg,gamma,b.CAD_S_END):
            record.update(ey_m=ey,heading_deg=deg)
            summaries.append(record)
    comparison=pd.DataFrame(summaries)
    comparison.to_csv(root/"comparison.csv",index=False)
    manifest={
        "type":"TRAIN_ONLY_FROZEN_ACTOR_CRITIC_COMPARISON",
        "train_sha256":digest,
        "gamma":gamma,
        "Qe_fixed":Q_EVAL_CANDIDATE.tolist(),
        "Re_fixed":R_EVAL_CANDIDATE.tolist(),
        "checkpoint_file":str(Path(args.checkpoint).resolve()),
        "checkpoint_episode":config.get("checkpoint_episode"),
        "max_steps":args.max_steps,
        "conditions":[{"ey_m":ey,"heading_deg":hdg} for ey,hdg in conditions],
        "screening_comparators":list(BASELINES),
        "evaluation_is_train_only":True,
        "actor_sampling":False,
        "actor_and_critic_parameter_updates":False,
        "no_test_used":True,
        "terminal_conditions_certified":False,
        "caution":"Single seed/checkpoint and TRAIN conditions are exploratory, NOT held-out TEST generalization; positive terminal P box is not mathematically stability certified.",
    }
    (root/"eval_config.json").write_text(
        json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"evaluation":"TRAIN_ONLY_FROZEN",
                      "rows":len(comparison),
                      "comparison_file":str(root/"comparison.csv"),
                      "test_used":False}),flush=True)


if __name__=="__main__":
    main()
