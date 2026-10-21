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
from direct_p_critic import SixParameterTDCritic
from evaluate_frozen_actor_critic import load_frozen_agents, frozen_rollout
from experiment_plots import (plot_episode_weights,plot_trajectories,
                              plot_frozen_comparison)
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
    if not cfg.EVALUATE_SAME_TRAIN_TRACK:
        raise ValueError("This run requires a frozen evaluation to compare learning")
    if cfg.REWARD_CANDIDATE!="legacy":
        raise ValueError("Unknown fixed reward candidate")
    resolved=resolve_train_reference(
        cfg.TRAIN_CSV,source=cfg.TRAIN_CSV_SOURCE)
    _, digest=verify_train_reference(resolved)
    return checkpoints,digest


def train_argv(checkpoints):
    """Build args only INSIDE Python; user runs without flags."""
    return [
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
        "--actor-output-scales",*(str(x) for x in cfg.ACTOR_OUTPUT_LR_FACTORS),
        "--critic-component-scales",*(str(x) for x in cfg.CRITIC_P_LR_FACTORS),
        "--episodes",str(cfg.EPISODES),
        "--steps-per-episode",str(cfg.STEPS_PER_EPISODE),
        "--seed",str(cfg.SEED),
        "--ey",str(cfg.INITIAL_EY_M),
        "--heading-deg",str(cfg.INITIAL_HEADING_DEG),
        "--save-at",*(str(x) for x in checkpoints),
    ]


def evaluate_checkpoints(checkpoints):
    """Evaluates matching frozen policies only; never selects using TEST."""
    low,high=screening_box(cfg.WEIGHT_BOX_MARGIN)
    q0,r0,p0=LEARNING_INITIAL.arrays()
    good_q,good_r,good_p=GOOD_NMPC.arrays()
    env=TrainEnvironment(
        train_length_m=b.CAD_S_END,q_initial=q0,r_initial=r0,p_initial=p0,
        task_qe=Q_EVAL_CANDIDATE,task_re=R_EVAL_CANDIDATE,
        max_steps=cfg.STEPS_PER_EPISODE,
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
    for ep in checkpoints:
        ckpt=cfg.RESULTS_DIR/f"checkpoint_ep{ep-1:04d}.npz"
        _,(actor,critic),meta=load_frozen_agents(
            ckpt,low,high,expected_gamma=cfg.GAMMA)
        policies.append((f"trained_episode_{ep:04d}",actor,critic))
    summaries=[]
    traces={}
    for name,actor,critic in policies:
        result,log=frozen_rollout(env,actor,critic,cfg.GAMMA,name)
        log.to_csv(cfg.RESULTS_DIR/f"eval_{name}.csv",index=False)
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
    frame.to_csv(cfg.RESULTS_DIR/"comparison.csv",index=False)
    return frame,traces


def main():
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
    comparison,traces=evaluate_checkpoints(checkpoints)
    cfg.IMAGES_DIR.mkdir(parents=True,exist_ok=True)
    figs=plot_episode_weights(history,cfg.IMAGES_DIR)
    figs["tracking"]=plot_trajectories(cfg.TRAIN_CSV,traces,cfg.IMAGES_DIR)
    figs["frozen_cte"]=plot_frozen_comparison(comparison,cfg.IMAGES_DIR)
    report={
        "status":"TRAIN_RUN_AND_FROZEN_EVALUATION_COMPLETE",
        "data_role":"TRAIN_ONLY","train_sha256":digest,
        "gamma":cfg.GAMMA, "episodes":cfg.EPISODES,
        "actor_output_lr_factors":list(cfg.ACTOR_OUTPUT_LR_FACTORS),
        "critic_p_lr_factors":list(cfg.CRITIC_P_LR_FACTORS),
        "checkpoints_1based":list(checkpoints),
        "fixed_reward_Qe":Q_EVAL_CANDIDATE.tolist(),
        "fixed_reward_Re":R_EVAL_CANDIDATE.tolist(),
        "experimental_QRP_domain_not_stability_certified":True,
        "no_test_used":True,
        "history_file":str(cfg.RESULTS_DIR/"training_history.csv"),
        "comparison_file":str(cfg.RESULTS_DIR/"comparison.csv"),
        "figures":figs,
        "note":"Images from actual frozen vehicle XY and logged Q/R/P, not generated estimates. One seed, one TRAIN path, one initial condition; no generalization conclusion.",
    }
    (cfg.RESULTS_DIR/"experiment_summary.json").write_text(
        json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report),flush=True)
    return report


if __name__=="__main__":
    main()
