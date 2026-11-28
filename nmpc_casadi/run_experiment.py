"""One command: python run_experiment.py

All user-editable settings are in experiment_config.py, NOT PowerShell.
Uses only the existing reference_split TRAIN trajectory, writes flat
numerical files into results/ and all figures into images/.

The main training loop still uses the approved 17-input two-head
Gaussian Actor, 6-parameter Direct-P TD(0) Critic, NMPC and fixed
Eq. (4) reward. Prior 27-case screen and reward audit are NOT read
from ad-hoc user paths by this simplified runner; the source-defined
20%-margin experimental Q/R/P box and Qe/Re are recorded in run_config.

Never creates or overwrites TEST data. Frozen evaluation reruns three
fixed controllers using the same adaptive-cost numerical NMPC,
rather than requiring any historical screening CSVs. Keep earlier
results_screen/results_rl material unchanged; these are archival.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Set BEFORE importing baseline: the previous standalone NMPC scripts retain
# their old sweep_results outputs, while this runner creates no such folders.
os.environ["NMPC_RL_MINIMAL_LAYOUT"]="1"
import baseline as b
import experiment_config as cfg
from audit_training_reward import Q_EVAL_CANDIDATE, R_EVAL_CANDIDATE
from best_episode_selection import select_best_frozen_checkpoint
from direct_p_critic import SixParameterTDCritic
from evaluate_frozen_actor_critic import load_frozen_agents, frozen_rollout
from experiment_plots import (plot_episode_weights,plot_trajectories,
                              plot_frozen_comparison,plot_best_train_episode,
                              plot_best_frozen_episode)
from screen_training_weights import verify_train_reference
from train_reference_locator import resolve_train_reference
from train_actor_critic import main as train_main
from train_rl_environment import TrainEnvironment
from training_initialization import (
    LEARNING_INITIAL, GOOD_NMPC, screening_box,
)


class FixedWeightActor:
    """Frozen numerical comparator: Q/R only; zero trainable parameters."""
    def __init__(self,q,r):
        self.q=np.asarray(q,float).copy()
        self.r=np.asarray(r,float).copy()
        if self.q.shape!=(6,) or self.r.shape!=(2,):
            raise ValueError("Incorrect fixed-controller Q/R dimensions")
        self.params={}

    def propose(self,observation,*,deterministic=False):
        if not deterministic:
            raise ValueError("Fixed NMPC has no stochastic Actor")
        return SimpleNamespace(
            q_diagonal=self.q.copy(),r_diagonal=self.r.copy(),
            sigma=np.zeros(8,dtype=float))


def _fixed_crit(p,lower,upper):
    return SixParameterTDCritic(p,lower,upper)


def _resolve_track_settings():
    """Auto-detect CAD track (non-pinned CSV) and steps per episode.

    User only edits TRAIN_CSV / RESULTS_DIR / IMAGES_DIR in experiment_config.
    STEPS_PER_EPISODE="auto" (or None) -> track length / (mean vx_ref * Ts) * 1.05.
    """
    from cad_eval_reference import sha256_file
    from training_initialization import TRAIN_REFERENCE_SHA256

    def auto_steps(csv):
        ref=pd.read_csv(csv)
        length=float(ref["s_m"].iloc[-1])
        speed=float(ref["vx_ref_mps"].mean())
        return int(np.ceil(length/(speed*float(b.Ts))*1.05))

    csv=Path(cfg.TRAIN_CSV)
    if not csv.is_file():
        raise FileNotFoundError(f"TRAIN_CSV not found: {csv}")
    cfg.ALLOW_CAD_REFERENCE=(sha256_file(csv)!=TRAIN_REFERENCE_SHA256)
    if getattr(cfg,"STEPS_PER_EPISODE",None) in (None,"auto"):
        cfg.STEPS_PER_EPISODE=auto_steps(csv)
    info={"track_A":csv.name,"steps_per_episode_A":cfg.STEPS_PER_EPISODE}
    csv_b=getattr(cfg,"TRAIN_CSV_B",None)
    if csv_b is not None:
        csv_b=Path(csv_b)
        if not csv_b.is_file():
            raise FileNotFoundError(f"TRAIN_CSV_B not found: {csv_b}")
        cfg.TRAIN_CSV_B=csv_b
        cfg.ALLOW_CAD_REFERENCE=True
        if getattr(cfg,"STEPS_PER_EPISODE_B","auto") in (None,"auto"):
            cfg.STEPS_PER_EPISODE_B=auto_steps(csv_b)
        if getattr(cfg,"SWITCH_EPISODE","half") in (None,"half"):
            cfg.SWITCH_EPISODE=cfg.EPISODES//2+1
        info.update({"track_B":csv_b.name,"steps_per_episode_B":cfg.STEPS_PER_EPISODE_B,
                     "episodes_A":f"1..{cfg.SWITCH_EPISODE-1}",
                     "episodes_B":f"{cfg.SWITCH_EPISODE}..{cfg.EPISODES}"})
    print(json.dumps(info),flush=True)


def eval_tracks():
    """(name, csv, steps, results_dir, images_dir) for each frozen evaluation."""
    csv_b=getattr(cfg,"TRAIN_CSV_B",None)
    if csv_b is None:
        return [("A",Path(cfg.TRAIN_CSV),cfg.STEPS_PER_EPISODE,
                 cfg.RESULTS_DIR,cfg.IMAGES_DIR)]
    out=[]
    for csv,steps in ((Path(cfg.TRAIN_CSV),cfg.STEPS_PER_EPISODE),
                      (Path(csv_b),cfg.STEPS_PER_EPISODE_B)):
        name=csv.stem.replace("_xref","")
        out.append((name,csv,steps,cfg.RESULTS_DIR/f"eval_{name}",
                    cfg.IMAGES_DIR/name))
    return out


def activate_eval_track(csv,results_dir):
    """Stage a track and point NMPC reference globals at it for evaluation."""
    from cad_eval_reference import prepare_cad_evaluation_reference
    ref_info,staged=prepare_cad_evaluation_reference(
        csv,results_dir,b.TAIL_LENGTH_M,
        min_speed_mps=float(b.cfg["vx_min"]),
        max_speed_mps=min(float(b.TEST_SPEED_MPS),float(b.cfg["vx_max"])),
        wheelbase_m=float(b.WHEELBASE_M),
        steering_limit_rad=min(abs(float(b.cfg["u_min"][1])),
                               abs(float(b.cfg["u_max"][1]))))
    b.REF_INFO=ref_info
    b.TAIL_FILE=Path(staged["tail_file"])
    b.CAD_S_END=float(ref_info["cad_s_end"])


def _validate_config():
    if not (1<=cfg.EPISODES and 1<=cfg.STEPS_PER_EPISODE):
        raise ValueError("Episode and step counts must be positive")
    checkpoints=tuple(cfg.CHECKPOINT_EPISODES)
    if (not checkpoints or tuple(sorted(set(checkpoints)))!=checkpoints or
        any(ep<1 or ep>cfg.EPISODES for ep in checkpoints)):
        raise ValueError("Selected checkpoint episodes must be ordered, unique and valid")
    for dirpath in (cfg.RESULTS_DIR,cfg.IMAGES_DIR):
        if dirpath.exists() and any(dirpath.iterdir()):
            raise FileExistsError(
                f"Existing {dirpath} is not empty. Preserve prior experiment. "
                "Move it to an archive or change the root path IN experiment_config.py")
    for name,n in (("ACTOR_OUTPUT_LR_FACTORS",8),("CRITIC_P_LR_FACTORS",6)):
        factors=np.asarray(getattr(cfg,name),dtype=float)
        if (factors.shape!=(n,) or not np.all(np.isfinite(factors))
            or np.any(factors<=0)):
            raise ValueError(f"{name} must contain {n} positive finite entries")
    for name in ("SOLVER_FAILURE_PENALTY","FAILURE_TERMINAL_PENALTY",
                 "ACTOR_MU_REG_COEF"):
        value=float(getattr(cfg,name,0.))
        if not np.isfinite(value) or value<0:
            raise ValueError(f"{name} must be nonnegative finite")
    threshold=float(getattr(cfg,"ACTOR_MU_REG_THRESHOLD",2.))
    if not np.isfinite(threshold) or threshold<=0.:
        raise ValueError("ACTOR_MU_REG_THRESHOLD must be strictly positive finite")
    if not cfg.EVALUATE_SAME_TRAIN_TRACK:
        raise ValueError("This run requires a frozen evaluation to compare learning")
    if cfg.REWARD_CANDIDATE!="legacy":
        raise ValueError("Unknown fixed reward candidate")
    if getattr(cfg,"ALLOW_CAD_REFERENCE",False):
        from cad_eval_reference import sha256_file
        if not Path(cfg.TRAIN_CSV).is_file():
            raise FileNotFoundError(f"TRAIN_CSV not found: {cfg.TRAIN_CSV}")
        return checkpoints,sha256_file(cfg.TRAIN_CSV)
    resolved=resolve_train_reference(
        cfg.TRAIN_CSV,source=cfg.TRAIN_CSV_SOURCE)
    _, digest=verify_train_reference(resolved)
    return checkpoints,digest


def train_argv(checkpoints):
    """Build args only INSIDE Python; user runs without flags."""
    cad=["--allow-cad-reference"] if getattr(cfg,"ALLOW_CAD_REFERENCE",False) else []
    if getattr(cfg,"TRAIN_CSV_B",None) is not None:
        cad+=["--reference-b",str(cfg.TRAIN_CSV_B),
              "--switch-episode",str(cfg.SWITCH_EPISODE),
              "--steps-per-episode-b",str(cfg.STEPS_PER_EPISODE_B)]
    return [*cad,
        "--reference",str(cfg.TRAIN_CSV),
        "--use-internal-box","--box-margin",str(cfg.WEIGHT_BOX_MARGIN),
        "--output",str(cfg.RESULTS_DIR),
        "--acknowledge-empirical-only",
        "--reward-candidate",cfg.REWARD_CANDIDATE,
        "--gamma",str(cfg.GAMMA),
        "--sigma-lower",str(cfg.SIGMA_MIN),
        "--sigma-initial",str(cfg.SIGMA_INIT),
        "--sigma-upper",str(cfg.SIGMA_MAX),
        "--actor-lr",str(cfg.ACTOR_LR),
        "--critic-lr",str(cfg.CRITIC_LR),
        "--critic-update-mode",getattr(cfg,"CRITIC_UPDATE_MODE","semi_gradient"),
        "--solver-failure-penalty",str(getattr(cfg,"SOLVER_FAILURE_PENALTY",0.)),
        "--failure-terminal-penalty",str(getattr(cfg,"FAILURE_TERMINAL_PENALTY",0.)),
        "--actor-mu-reg-coef",str(getattr(cfg,"ACTOR_MU_REG_COEF",0.)),
        "--actor-mu-reg-threshold",str(getattr(cfg,"ACTOR_MU_REG_THRESHOLD",2.)),
        "--actor-output-scales",*(str(x) for x in cfg.ACTOR_OUTPUT_LR_FACTORS),
        "--critic-component-scales",*(str(x) for x in cfg.CRITIC_P_LR_FACTORS),
        "--episodes",str(cfg.EPISODES),
        "--steps-per-episode",str(cfg.STEPS_PER_EPISODE),
        "--seed",str(cfg.SEED),
        "--ey",str(cfg.INITIAL_EY_M),
        "--heading-deg",str(cfg.INITIAL_HEADING_DEG),
        "--save-at",*(str(x) for x in checkpoints),
    ]


def evaluate_checkpoints(checkpoints,out_dir=None,max_steps=None):
    """Evaluates matching frozen policies only; never selects using TEST.

    Checkpoints are read from RESULTS_DIR; evaluation files go to out_dir.
    """
    out_dir=Path(cfg.RESULTS_DIR if out_dir is None else out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    max_steps=cfg.STEPS_PER_EPISODE if max_steps is None else max_steps
    low,high=screening_box(cfg.WEIGHT_BOX_MARGIN)
    q0,r0,p0=LEARNING_INITIAL.arrays()
    good_q,good_r,good_p=GOOD_NMPC.arrays()
    env=TrainEnvironment(
        train_length_m=b.CAD_S_END,q_initial=q0,r_initial=r0,p_initial=p0,
        task_qe=Q_EVAL_CANDIDATE,task_re=R_EVAL_CANDIDATE,
        max_steps=max_steps,
        ey=cfg.INITIAL_EY_M,heading_deg=cfg.INITIAL_HEADING_DEG,
    )
    # Evaluate once at baseline and untrained initial policy.
    first_ckpt=cfg.RESULTS_DIR/f"checkpoint_ep{checkpoints[0]-1:04d}.npz"
    (untrained,c_initial),_,meta=load_frozen_agents(
        first_ckpt,low,high,expected_gamma=cfg.GAMMA)
    policies=[
        ("untrained_deterministic",untrained,c_initial),
        ("learning_initial",FixedWeightActor(q0,r0),_fixed_crit(p0,low[2],high[2])),
        ("strong_fixed_baseline",FixedWeightActor(good_q,good_r),
            _fixed_crit(good_p,low[2],high[2])),
        ("stage_lower_terminal_upper",FixedWeightActor(low[0],low[1]),
            _fixed_crit(high[2],low[2],high[2])),
    ]
    checkpoint_sources={}
    for ep in checkpoints:
        ckpt=cfg.RESULTS_DIR/f"checkpoint_ep{ep-1:04d}.npz"
        _,(actor,critic),meta=load_frozen_agents(
            ckpt,low,high,expected_gamma=cfg.GAMMA)
        label=f"trained_episode_{ep:04d}"
        policies.append((label,actor,critic))
        checkpoint_sources[label]=ckpt
    best_train_manifest=cfg.RESULTS_DIR/"best_train_selection.json"
    if best_train_manifest.is_file():
        train_info=json.loads(best_train_manifest.read_text(encoding="utf-8"))
        best_ep=int(train_info["episode_1based"])
        if not 1<=best_ep<=cfg.EPISODES:
            raise ValueError("Best TRAIN checkpoint has invalid episode")
        label=f"trained_episode_{best_ep:04d}"
        if label not in checkpoint_sources:
            ckpt=cfg.RESULTS_DIR/"checkpoint_best_train.npz"
            _,(actor,critic),meta=load_frozen_agents(
                ckpt,low,high,expected_gamma=cfg.GAMMA)
            if int(meta["checkpoint_episode"]) != best_ep-1:
                raise ValueError("Best TRAIN checkpoint metadata episode mismatch")
            policies.append((label,actor,critic))
            checkpoint_sources[label]=ckpt
    summaries=[]
    traces={}
    for name,actor,critic in policies:
        result,log=frozen_rollout(env,actor,critic,cfg.GAMMA,name)
        log.to_csv(out_dir/f"eval_{name}.csv",index=False)
        summaries.append(result)
        traces[name]=log
        print(json.dumps({
            "frozen_controller":name,
            "mean_cte_post_m":result["mean_cte_post_m"],
            "discounted_cost_from_start":result["discounted_cost_from_start"],
            "reached_end":result["reached_end"],
            "solver_success_pct":result["solver_success_pct"],
        }),flush=True)
    frame=pd.DataFrame(summaries)
    frame.to_csv(out_dir/"comparison.csv",index=False)
    best=select_best_frozen_checkpoint(summaries)
    if best is not None:
        label=best["controller"]
        episode=int(label.rsplit("_",1)[1])
        ckpt=checkpoint_sources[label]
        # Clear names make best policy usable without guessing epochs.
        shutil.copy2(ckpt,out_dir/"checkpoint_best_frozen.npz")
        traces[label].to_csv(out_dir/"eval_best_frozen.csv",index=False)
        frozen_info={
            "type":"BEST_DETERMINISTIC_FROZEN_TRAIN",
            "episode_1based":episode,
            "controller":label,
            "criterion":"completed_and_solver_success_at_least_99_pct_then_min_post_1p3m_cte",
            "mean_cte_post_m":float(best["mean_cte_post_m"]),
            "max_cte_post_m":float(best["max_cte_post_m"]),
            "mean_reward_paper":float(best["mean_reward"]),
            "solver_success_pct":float(best["solver_success_pct"]),
            "completion_pct":float(best["completion_pct"]),
            "candidate_episodes_evaluated":sorted(int(x.rsplit("_",1)[1])
                for x in checkpoint_sources),
            "not_global_best_over_all_episodes":True,
            "reference_role":"TRAIN_ONLY",
            "uses_TEST":False,
        }
    else:
        frozen_info={
            "type":"NO_QUALIFYING_FROZEN_CHECKPOINT",
            "criterion":"requires_completed_reached_end_and_solver_success_at_least_99_pct",
            "candidate_episodes_evaluated":sorted(int(x.rsplit("_",1)[1])
                for x in checkpoint_sources),
            "reference_role":"TRAIN_ONLY","uses_TEST":False,
        }
    (out_dir/"best_frozen_selection.json").write_text(
        json.dumps(frozen_info,indent=2),encoding="utf-8")
    return frame,traces


def main():
    _resolve_track_settings()
    checkpoints,digest=_validate_config()
    print(json.dumps({
        "stage":"TRAIN", "episodes":cfg.EPISODES, "gamma":cfg.GAMMA,
        "trajectory":str(cfg.TRAIN_CSV), "results":str(cfg.RESULTS_DIR),
        "images":str(cfg.IMAGES_DIR), "checkpoint_episodes":checkpoints,
        "train_sha256":digest,
        "warning":"Fixed-screen box is exploratory; terminal stability not mathematically proven",
    }),flush=True)
    history=train_main(train_argv(checkpoints))
    if len(history)!=cfg.EPISODES:
        raise RuntimeError("Training ended early without all requested episodes")
    if any(not np.isfinite(row["mean_reward"]) for row in history):
        raise FloatingPointError("Training produced nonfinite reward")
    print("All episodes finished; freezing weights for deterministic TRAIN evaluation.",flush=True)
    cfg.IMAGES_DIR.mkdir(parents=True,exist_ok=True)
    figs=plot_episode_weights(history,cfg.IMAGES_DIR)
    best_train_path=cfg.RESULTS_DIR/"best_train_selection.json"
    train_best=(json.loads(best_train_path.read_text(encoding="utf-8"))
                if best_train_path.is_file() else None)
    if train_best is not None:
        figs["best_train"]=plot_best_train_episode(
            pd.read_csv(cfg.RESULTS_DIR/"episode_best_train.csv"),
            train_best["episode_1based"],cfg.IMAGES_DIR)
    tracks=eval_tracks()
    curriculum=len(tracks)>1
    frozen_by_track={}
    online_by_track={}
    for name,csv,steps,res_dir,img_dir in tracks:
        # Frozen evaluation of the SAME checkpoints on every training track.
        print(json.dumps({"stage":"FROZEN_EVALUATION","track":name,
                          "results":str(res_dir),"images":str(img_dir)}),flush=True)
        if curriculum:
            activate_eval_track(csv,res_dir)
        comparison,traces=evaluate_checkpoints(checkpoints,res_dir,steps)
        img_dir.mkdir(parents=True,exist_ok=True)
        key=(lambda k:f"{name}_{k}") if curriculum else (lambda k:k)
        figs[key("tracking")]=plot_trajectories(csv,traces,img_dir)
        figs[key("frozen_cte")]=plot_frozen_comparison(comparison,img_dir)
        frozen_best=json.loads(
            (res_dir/"best_frozen_selection.json").read_text(encoding="utf-8"))
        if frozen_best["type"]=="BEST_DETERMINISTIC_FROZEN_TRAIN":
            selected=frozen_best["controller"]
            best_figs=plot_best_frozen_episode(
                csv,traces[selected],selected,img_dir,
                comparator_trace=traces.get("strong_fixed_baseline"))
            figs.update({key(f"best_frozen_{k}"):v for k,v in best_figs.items()})
        frozen_by_track[name]=frozen_best
        online_summary=None
        if getattr(cfg,"RUN_ONLINE_ADAPTATION",False):
            from online_adaptation import run_from_checkpoint
            checkpoint=res_dir/"checkpoint_best_frozen.npz"
            if not checkpoint.is_file():
                checkpoint=cfg.RESULTS_DIR/"checkpoint_best_train.npz"
            if checkpoint.is_file():
                q0,r0,p0=LEARNING_INITIAL.arrays()
                online_env=TrainEnvironment(
                    train_length_m=b.CAD_S_END,q_initial=q0,r_initial=r0,p_initial=p0,
                    task_qe=Q_EVAL_CANDIDATE,task_re=R_EVAL_CANDIDATE,
                    max_steps=steps,ey=cfg.INITIAL_EY_M,
                    heading_deg=cfg.INITIAL_HEADING_DEG)
                online_summary=run_from_checkpoint(online_env,checkpoint,
                    res_dir/"online_adaptation",
                    seed=getattr(cfg,"ONLINE_ADAPTATION_SEED",cfg.SEED))
                print(json.dumps(online_summary),flush=True)
            else:
                online_summary={"status":"SKIPPED_NO_QUALIFYING_CHECKPOINT"}
        online_by_track[name]=online_summary
    if curriculum:
        frozen_best,online_summary=frozen_by_track,online_by_track
    else:
        frozen_best=frozen_by_track["A"]
        online_summary=online_by_track["A"]
    report={
        "online_adaptation":online_summary,
        "status":"TRAIN_RUN_AND_FROZEN_EVALUATION_COMPLETE",
        "data_role":"TRAIN_ONLY","train_sha256":digest,
        "gamma":cfg.GAMMA, "episodes":cfg.EPISODES,
        "experiment_variant":"D_training_safety_plus_actor_mu_regularization",
        "train_solver_failure_penalty":getattr(cfg,"SOLVER_FAILURE_PENALTY",0.),
        "train_failure_terminal_penalty":getattr(cfg,"FAILURE_TERMINAL_PENALTY",0.),
        "actor_mu_reg_coef":getattr(cfg,"ACTOR_MU_REG_COEF",0.),
        "actor_mu_reg_threshold":getattr(cfg,"ACTOR_MU_REG_THRESHOLD",2.),
        "evaluation_reward_type":"paper_eq4_unmodified",
        "actor_output_lr_factors":list(cfg.ACTOR_OUTPUT_LR_FACTORS),
        "critic_p_lr_factors":list(cfg.CRITIC_P_LR_FACTORS),
        "critic_update_mode":getattr(cfg,"CRITIC_UPDATE_MODE","semi_gradient"),
        "checkpoints_1based":list(checkpoints),
        "best_train_selection":train_best,
        "best_frozen_selection":frozen_best,
        "fixed_reward_Qe":Q_EVAL_CANDIDATE.tolist(),
        "fixed_reward_Re":R_EVAL_CANDIDATE.tolist(),
        "experimental_QRP_domain_not_stability_certified":True,
        "no_test_used":True,
        "history_file":str(cfg.RESULTS_DIR/"training_history.csv"),
        "comparison_file":(str(cfg.RESULTS_DIR/"comparison.csv") if not curriculum
            else {n:str(r/"comparison.csv") for n,_,_,r,_ in tracks}),
        "curriculum":(None if not curriculum else {
            "track_A":str(cfg.TRAIN_CSV),"track_B":str(cfg.TRAIN_CSV_B),
            "switch_episode_1based":cfg.SWITCH_EPISODE}),
        "figures":figs,
        "note":"Images from actual frozen vehicle XY and logged Q/R/P, not generated estimates. One seed, one TRAIN path, one initial condition; no generalization conclusion.",
    }
    (cfg.RESULTS_DIR/"experiment_summary.json").write_text(
        json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report),flush=True)
    return report


if __name__=="__main__":
    main()
