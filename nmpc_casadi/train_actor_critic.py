"""GATED first joint Gaussian-Actor / direct-P Critic / NMPC TRAIN driver.

No implicit learning rates, discount, or Gaussian dispersion limits.
No TEST path accepted. Qe/Re are fixed OLD-code reward CANDIDATES
ONLY when the caller explicitly selects --reward-candidate legacy.
Box must come from the pinned TRAIN full-screening manifest AND caller
must acknowledge that finite empirical screening is NOT a proof of
terminal-set invariance or stability under adaptive weights.

Start with e.g. 5 or 10 steps as a numerical diagnostic. This script
does NOT silently run a long experiment or tune on TEST.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

import baseline as b
from audit_training_reward import Q_EVAL_CANDIDATE, R_EVAL_CANDIDATE
from actor_critic_updates import learn_one_transition
from experiment_d_penalties import compute_training_reward
from cad_eval_reference import prepare_cad_evaluation_reference
from gaussian_weight_actor import GaussianWeightActor, actor_mu_excess_squared
from direct_p_critic import SixParameterTDCritic
from rl_paper_core import td_error
from screen_training_weights import verify_train_reference
from train_rl_environment import TrainEnvironment
from training_initialization import LEARNING_INITIAL, TRAIN_REFERENCE_SHA256, screening_box


def read_screened_box(path):
    doc=json.loads(Path(path).read_text(encoding="utf-8"))
    if (doc.get("train_sha256") != TRAIN_REFERENCE_SHA256 or
        doc.get("profile")!="full" or
        doc.get("empirical_all_pass") is not True or
        doc.get("total_samples")!=27 or doc.get("pass_samples")!=27):
        raise ValueError("Require full 27/27 PASS on pinned TRAIN reference")
    lo=tuple(np.asarray(v,float) for v in doc["screen_candidate_lower"])
    hi=tuple(np.asarray(v,float) for v in doc["screen_candidate_upper"])
    if ([x.shape for x in lo]!=[(6,),(2,),(6,)] or
        [x.shape for x in hi]!=[(6,),(2,),(6,)]):
        raise ValueError("Incorrect screened Q/R/P array dimensions")
    init=LEARNING_INITIAL.arrays()
    for a,b,v in zip(lo,hi,init):
        if (np.any(a<=0) or np.any(a>=v) or np.any(v>=b) or
            not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))):
            raise ValueError("Nominal initial weights not inside screening box")
    return lo,hi,doc


def verify_reward_audit(path, gamma):
    doc=json.loads(Path(path).read_text(encoding="utf-8"))
    if (doc.get("type")!="TRAIN_ONLY_FIXED_REWARD_CANDIDATE_NO_LEARNING" or
        doc.get("train_sha256")!=TRAIN_REFERENCE_SHA256 or
        doc.get("cases")!=27 or
        doc.get("Qe")!=Q_EVAL_CANDIDATE.tolist() or
        doc.get("Re")!=R_EVAL_CANDIDATE.tolist() or
        not np.isclose(doc.get("gamma_diagnostic",float("nan")),gamma)):
        raise ValueError("Require 27-case pinned TRAIN reward/TD audit with the SAME gamma and Qe/Re")
    return doc


def write_checkpoint(path,actor,critic,metadata):
    arrays={f"actor_{name}":v for name,v in actor.params.items()}
    arrays["critic_physical_P"]=critic.p
    arrays["metadata"]=np.array(json.dumps(metadata))
    np.savez_compressed(path,**arrays)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference",type=Path,required=True)
    ap.add_argument("--screen-manifest",type=Path)
    ap.add_argument("--use-internal-box",action="store_true",
                    help="Use source-defined, formerly screened 20-percent experimental bounds; no external manifest needed")
    ap.add_argument("--box-margin",type=float,default=0.20)
    ap.add_argument("--save-at",type=int,nargs="+",
                    help="Save detailed episode CSV and checkpoint only after these 1-based episodes")
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--reward-audit-summary",type=Path,
                    help="Mandatory for >10 steps or >1 episode: checked 27-case TRAIN reward audit summary")
    ap.add_argument("--acknowledge-empirical-only",action="store_true",
                    help="Acknowledge 27/27 fixed-weight runs do NOT prove closed-loop stability for adaptive Q/R/P")
    ap.add_argument("--reward-candidate",choices=["legacy"],required=True,
                    help="Explicitly accept the OLD code's fixed Qe/Re as a diagnostic candidate")
    ap.add_argument("--gamma",type=float,required=True)
    ap.add_argument("--sigma-lower",type=float,required=True)
    ap.add_argument("--sigma-initial",type=float,required=True)
    ap.add_argument("--sigma-upper",type=float,required=True)
    ap.add_argument("--actor-lr",type=float,required=True)
    ap.add_argument("--critic-lr",type=float,required=True)
    ap.add_argument("--actor-output-scales",type=float,nargs=8,
                    metavar=("Qvx","Qvy","Qpsi","Qr","QX","QY","Ra","Rdelta"))
    ap.add_argument("--critic-component-scales",type=float,nargs=6,
                    metavar=("Pvx","Pvy","Ppsi","Pr","PX","PY"))
    ap.add_argument("--solver-failure-penalty",type=float,default=0.,
                    help="TRAIN-only penalty after BOTH NMPC solve attempts fail")
    ap.add_argument("--failure-terminal-penalty",type=float,default=0.,
                    help="TRAIN-only catastrophic terminal penalty")
    ap.add_argument("--actor-mu-reg-coef",type=float,default=0.,
                    help="Actor-only latent-mean squared-hinge loss coefficient")
    ap.add_argument("--actor-mu-reg-threshold",type=float,default=2.,
                    help="Latent-mean absolute value threshold for regularization")
    ap.add_argument("--episodes",type=int,required=True)
    ap.add_argument("--steps-per-episode",type=int,required=True)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--ey",type=float,default=.05)
    ap.add_argument("--heading-deg",type=float,default=0.)
    args=ap.parse_args(argv)

    if not args.acknowledge_empirical_only:
        ap.error("Explicit --acknowledge-empirical-only required before adaptive-weight simulation")
    if args.episodes<1 or args.steps_per_episode<1:
        ap.error("episodes and steps-per-episode must be positive")
    if not np.isfinite(args.gamma) or not 0<args.gamma<=1.:
        ap.error("gamma must be in (0,1]")
    for k in ("actor_lr","critic_lr"):
        v=getattr(args,k)
        if not np.isfinite(v) or v<=0:
            ap.error(f"{k} must be positive finite")
    for k in ("solver_failure_penalty","failure_terminal_penalty",
              "actor_mu_reg_coef"):
        v=getattr(args,k)
        if not np.isfinite(v) or v<0.:
            ap.error(f"{k} must be nonnegative finite")
    if not np.isfinite(args.actor_mu_reg_threshold) or args.actor_mu_reg_threshold<=0.:
        ap.error("actor_mu_reg_threshold must be strictly positive finite")
    if args.use_internal_box and args.screen_manifest is not None:
        ap.error("Choose either source-defined box or external screen manifest")
    if not args.use_internal_box and args.screen_manifest is None:
        ap.error("Either --use-internal-box or --screen-manifest is required")
    for name,values,length in (
        ("actor_output_scales",args.actor_output_scales,8),
        ("critic_component_scales",args.critic_component_scales,6),
    ):
        if values is not None and (
            len(values)!=length or
            not np.all(np.isfinite(values)) or
            np.any(np.asarray(values)<=0)
        ):
            ap.error(f"{name} must be {length} finite positive multipliers")
    if (args.episodes>1 or args.steps_per_episode>10) and args.reward_audit_summary is None and not args.use_internal_box:
        ap.error("Long CLI training with an external screening manifest still requires the matching reward audit")
    if args.save_at is not None and (len(set(args.save_at))!=len(args.save_at)
                                   or any(ep<1 or ep>args.episodes for ep in args.save_at)):
        ap.error("--save-at must contain unique 1-based episodes within the run")
    try:
        ref,digest=verify_train_reference(args.reference)
        if args.use_internal_box:
            lo,hi=screening_box(args.box_margin)
            screen_doc={
                "profile":"source_constant_screen_design",
                "total_samples":None,
                "empirical_all_pass":None,
            }
        else:
            lo,hi,screen_doc=read_screened_box(args.screen_manifest)
        reward_audit_doc=(
            verify_reward_audit(args.reward_audit_summary,args.gamma)
            if args.reward_audit_summary is not None else None
        )
    except (ValueError,KeyError,FileNotFoundError) as exc:
        ap.error(str(exc))
    out=args.output.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        ap.error("Output must be empty; never overwrite previous runs")
    out.mkdir(parents=True,exist_ok=True)

    ref_info,staged=prepare_cad_evaluation_reference(
        ref,out,b.TAIL_LENGTH_M,
        min_speed_mps=float(b.cfg["vx_min"]),
        max_speed_mps=min(float(b.TEST_SPEED_MPS),float(b.cfg["vx_max"])),
        wheelbase_m=float(b.WHEELBASE_M),
        steering_limit_rad=min(abs(float(b.cfg["u_min"][1])),
                               abs(float(b.cfg["u_max"][1]))))
    if staged["reference_sha256"]!=digest:
        raise RuntimeError("TRAIN reference staging changed the original")
    if args.use_internal_box:
        # Flat layout: put supporting trajectory derivatives alongside
        # original pinned TRAIN CSV, NOT in the numerical results folder.
        import shutil
        from experiment_config import TRAJECTORY_DIR
        targets={
            "reference_file":TRAJECTORY_DIR/"_rl_validated_train_copy.csv",
            "tail_file":TRAJECTORY_DIR/"_rl_prediction_tail.csv",
        }
        from cad_eval_reference import sha256_file
        for key,target in targets.items():
            source_derivative=Path(staged[key])
            if target.exists():
                # Only reuse byte-identical derived trajectories. Refuse to
                # overwrite user-modified files; remove only our OWN freshly
                # generated temporary copy, not prior experiment outputs.
                if sha256_file(target)!=sha256_file(source_derivative):
                    raise FileExistsError(
                        f"Existing derived TRAIN CSV differs: {target}")
                source_derivative.unlink()
            else:
                shutil.move(str(source_derivative),str(target))
            staged[key]=str(target)
    b.REF_INFO=ref_info
    b.TAIL_FILE=Path(staged["tail_file"])
    b.CAD_S_END=float(ref_info["cad_s_end"])
    q0,r0,p0=LEARNING_INITIAL.arrays()
    rng=np.random.default_rng(args.seed)
    actor=GaussianWeightActor(
        rng,q0,r0,lo[0],hi[0],lo[1],hi[1],
        sigma_lower=args.sigma_lower,sigma_upper=args.sigma_upper,
        sigma_initial=args.sigma_initial)
    critic=SixParameterTDCritic(p0,lo[2],hi[2])
    # Reward evaluator NEVER changes with Actor q/r.
    qe=Q_EVAL_CANDIDATE.copy()
    re=R_EVAL_CANDIDATE.copy()
    env=TrainEnvironment(
        train_length_m=b.CAD_S_END,q_initial=q0,r_initial=r0,p_initial=p0,
        task_qe=qe,task_re=re,max_steps=args.steps_per_episode,
        ey=args.ey,heading_deg=args.heading_deg)
    meta={
        "status":"EXPERIMENTAL_SIMULATION_ONLY_NOT_STABILITY_CERTIFIED",
        "reference_role":"TRAIN_ONLY","train_sha256":digest,
        "train_length_m":b.CAD_S_END,"Ts_s":b.Ts,"N":b.N,
        "initial_Q":q0.tolist(),"initial_R":r0.tolist(),
        "initial_P":p0.tolist(),
        "Q_lower":lo[0].tolist(),"Q_upper":hi[0].tolist(),
        "R_lower":lo[1].tolist(),"R_upper":hi[1].tolist(),
        "P_lower":lo[2].tolist(),"P_upper":hi[2].tolist(),
        "Qe_fixed":qe.tolist(),"Re_fixed":re.tolist(),
        "reward_type":("paper_eq4_plus_train_only_safety_penalties"
                       if args.solver_failure_penalty>0. or args.failure_terminal_penalty>0.
                       else "paper_eq4_no_failure_penalty_no_reward_shaping"),
        "evaluation_reward_type":"paper_eq4_only",
        "train_solver_failure_penalty":args.solver_failure_penalty,
        "train_failure_terminal_penalty":args.failure_terminal_penalty,
        "actor_mu_reg_coef":args.actor_mu_reg_coef,
        "actor_mu_reg_threshold":args.actor_mu_reg_threshold,
        "actor_arch":"17-64-64-tanh-learned_mu8_learned_sigma8",
        "critic_arch":"direct_six_physical_P_no_NN",
        "actor_optimizer":"SGD_paper_score_function",
        "critic_optimizer":"projected_TD0_semi_gradient",
        "actor_lr":args.actor_lr,"critic_lr":args.critic_lr,
        "actor_output_lr_factors": (
            list(args.actor_output_scales) if args.actor_output_scales is not None
            else [1.]*8
        ),
        "critic_p_lr_factors": (
            list(args.critic_component_scales) if args.critic_component_scales is not None
            else [1.]*6
        ),
        "actor_gradient_mode":"per_output_score_preconditioning_before_shared_backprop",
        "critic_gradient_mode":"per_component_projected_TD0",
        "gamma":args.gamma,
        "sigma_lower":args.sigma_lower,
        "sigma_initial":args.sigma_initial,
        "sigma_upper":args.sigma_upper,
        "seed":args.seed,"episodes":args.episodes,
        "steps_per_episode":args.steps_per_episode,
        "ey_m":args.ey,"heading_deg":args.heading_deg,
        "screen_profile":screen_doc["profile"],
        "reward_audit_verified":reward_audit_doc is not None,
        "weight_box_source":"source_defined_margin_only_not_stability_certified" if args.use_internal_box else "27_case_screening_manifest",
        "fixed_reward_source":"source_code_fixed_Qe_Re_not_external_audit" if reward_audit_doc is None else "verified_27_case_reward_audit",
        "offline_empirical_screen_count":screen_doc["total_samples"],
        "staging_detail":"Generic CAD reference-tail utility used only to stage TRAIN; no TEST inputs",
        "terminal_conditions_certified":False,
    }
    (out/"run_config.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    history=[]
    for ep in range(args.episodes):
        obs=env.reset()
        rows=[]
        for step in range(args.steps_per_episode):
            proposal=actor.propose(obs.features,deterministic=False)
            p_before=critic.p
            next_obs,reward,terminated,truncated,info=env.step(
                proposal.q_diagonal,proposal.r_diagonal,critic,obs)
            breakdown=compute_training_reward(
                reward,info,
                solver_failure_penalty=args.solver_failure_penalty,
                failure_terminal_penalty=args.failure_terminal_penalty)
            mu_excess_sq=float(actor_mu_excess_squared(
                proposal.mu,args.actor_mu_reg_threshold))
            update=learn_one_transition(
                actor,critic,obs.features,proposal.latent,
                obs.error,next_obs.error,breakdown.training_reward,
                gamma=args.gamma,actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,terminated=terminated,
                actor_output_scales=args.actor_output_scales,
                critic_component_scales=args.critic_component_scales,
                actor_mu_reg_coef=args.actor_mu_reg_coef,
                actor_mu_reg_threshold=args.actor_mu_reg_threshold)
            row={
                "episode":ep,"step":step,
                # reward always refers to the unchanged Eq.(4) task metric.
                "reward":reward,
                "training_reward":breakdown.training_reward,
                "penalty_solver":breakdown.solver_penalty,
                "penalty_terminal":breakdown.terminal_penalty,
                "actor_mu_excess_sq":mu_excess_sq,
                "actor_mu_regularization":args.actor_mu_reg_coef*mu_excess_sq,
                "actor_mu_beyond_threshold":int(np.sum(
                    np.abs(proposal.mu)>args.actor_mu_reg_threshold)),
                "td":update.td,
                "value_now":update.value_current,
                "value_next":update.value_next,
                "critic_loss":update.critic_loss,
                "actor_loss":update.actor_loss,
                "actor_grad_norm":update.actor_grad_norm,
                "actor_param_step_norm":update.actor_param_step_norm,
                "critic_param_step_norm":update.critic_param_step_norm,
                "terminated":terminated,"truncated":truncated,
                "sigma_mean":float(np.mean(proposal.sigma)),
                "sigma_min":float(np.min(proposal.sigma)),
                "sigma_max":float(np.max(proposal.sigma)),
                **info,
            }
            for name,vector in (
                ("q",proposal.q_diagonal),("r",proposal.r_diagonal),
                ("p_before",p_before),("p_after",critic.p),
                ("mu",proposal.mu),("sigma",proposal.sigma),
                ("latent",proposal.latent),
                ("e",obs.error),
            ):
                for i,value in enumerate(vector):
                    row[f"{name}_{i}"]=float(value)
            rows.append(row)
            obs=next_obs
            if terminated or truncated:
                break
        df=pd.DataFrame(rows)
        if args.save_at is None or (ep+1) in args.save_at:
            df.to_csv(out/f"episode_{ep:04d}.csv",index=False)
        record={
            "episode":ep,"steps":len(df),
            "return":float(df.reward.sum()),
            "mean_reward":float(df.reward.mean()),
            "training_return":float(df.training_reward.sum()),
            "mean_training_reward":float(df.training_reward.mean()),
            "penalty_solver_total":float(df.penalty_solver.sum()),
            "penalty_terminal_total":float(df.penalty_terminal.sum()),
            "actor_mu_regularization_mean":float(df.actor_mu_regularization.mean()),
            "actor_mu_excess_sq_mean":float(df.actor_mu_excess_sq.mean()),
            "actor_mu_fraction_beyond_threshold":float(
                df.actor_mu_beyond_threshold.mean()/8.),
            "rms_td":float(np.sqrt(np.mean(df.td**2))),
            "actor_grad_norm_max":float(df.actor_grad_norm.max()),
            "critic_step_norm_max":float(df.critic_param_step_norm.max()),
            "mean_cte_m":float(df.cte_m.mean()),
            "solver_success_pct":float(100*df.solver_success.mean()),
            "reached_end":bool(df.reached_end.iloc[-1]),
            "terminated":bool(df.terminated.iloc[-1]),
            "truncated":bool(df.truncated.iloc[-1]),
        }
        # Compact all-episode Q/R/Qf histories for plotting without 100
        # separate per-step CSVs. Means are sampled ACTION weights per
        # episode; Qf is the learned Critic P after the final TD update.
        for prefix,n in (("q",6),("r",2)):
            for i in range(n):
                record[f"{prefix}_mean_{i}"]=float(df[f"{prefix}_{i}"].mean())
                record[f"{prefix}_std_{i}"]=float(df[f"{prefix}_{i}"].std(ddof=0))
        for i,value in enumerate(critic.p):
            record[f"qf_final_{i}"]=float(value)
        record["sigma_mean_episode"]=float(df["sigma_mean"].mean())
        record["qf_step_total"]=float(df["critic_param_step_norm"].sum())
        history.append(record)
        pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
        if args.save_at is None or (ep+1) in args.save_at:
            write_checkpoint(out/f"checkpoint_ep{ep:04d}.npz",actor,critic,
                             {**meta,"checkpoint_episode":ep})
        print(json.dumps(record),flush=True)
    return history


if __name__ == "__main__":
    main()
