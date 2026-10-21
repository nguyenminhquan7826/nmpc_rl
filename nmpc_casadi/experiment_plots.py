"""Human-facing figures in ONE images directory, numerical data in results.

Figures distinguish stochastic Q/R episode means from deterministic
frozen replay. No plotting tool modifies the experiment parameters.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STATE_LABELS = ("vx", "vy", "psi", "r", "X", "Y")
ACTION_LABELS = ("acceleration", "steering")


def _save(fig, target):
    path=Path(target)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Do not overwrite existing figure: {path}")
    fig.savefig(path,dpi=180,bbox_inches="tight")
    plt.close(fig)
    return path


def plot_episode_weights(history, images_dir):
    """Three SEPARATE plots: actual sampled Actor Q/R mean and Critic Qf P."""
    df=pd.DataFrame(history)
    if df.empty:
        raise ValueError("No episodes")
    ep=df["episode"].to_numpy(int)+1
    results={}
    for key,n,names,title,unit in (
        ("q_mean",6,STATE_LABELS,"Sampled Actor Q (episode means)","Q diagonal"),
        ("r_mean",2,ACTION_LABELS,"Sampled Actor R (episode means)","R diagonal"),
        ("qf_final",6,STATE_LABELS,"Learned Critic Qf = diag(P) (episode end)","Qf diagonal"),
    ):
        fig,ax=plt.subplots(figsize=(10,5.2))
        for i in range(n):
            col=f"{key}_{i}"
            if col not in df:
                raise ValueError(f"Missing episode weight series {col}")
            ax.plot(ep,df[col].to_numpy(float),label=names[i],lw=1.45)
        ax.set(xlabel="Completed TRAIN episodes",ylabel=unit,title=title)
        ax.grid(alpha=.28)
        ax.legend(loc="best",ncol=3 if n==6 else 2,fontsize=8)
        results[key]=str(_save(fig,Path(images_dir)/f"{key}_episodes.png"))
    fig,ax=plt.subplots(figsize=(10,5.2))
    ax.plot(ep,df["mean_reward"],label="Mean paper reward / control step")
    ax.plot(ep,df["rms_td"],label="RMS TD(0) error")
    ax.set(xlabel="Completed TRAIN episodes",title="Learning diagnostics (TRAIN)")
    ax.grid(alpha=.25)
    ax.legend(loc="best")
    results["learning"]=str(_save(fig,Path(images_dir)/"training_reward_td.png"))
    return results


def plot_trajectories(reference, traces, images_dir):
    """Reference vs actual post-training/fixed-control XY, NOT a schematic."""
    if not traces:
        raise ValueError("No actual vehicle trajectories were provided")
    ref=pd.read_csv(reference)
    fig,ax=plt.subplots(figsize=(10,8))
    ax.plot(ref["X_ref_m"],ref["Y_ref_m"],color="black",
            lw=2.5,linestyle="--",label="Reference TRAIN")
    labels=list(traces)
    # Limit clutter while preserving final, initial and comparator curves.
    preferred=[
        "untrained_deterministic",
        "trained_episode_0001",
        "trained_episode_0025",
        "trained_episode_0100",
        "strong_fixed_baseline",
        "stage_lower_terminal_upper",
    ]
    selected=[x for x in preferred if x in traces]
    if not selected:
        selected=labels
    for label in selected:
        log=traces[label]
        if not {"X_m","Y_m"}.issubset(log.columns):
            raise ValueError(f"Missing true XY from {label}")
        ax.plot(log["X_m"],log["Y_m"],label=label,lw=1.25)
    ax.set(xlabel="Global X (m)",ylabel="Global Y (m)",
           title="Reference vs actual NMPC-controlled vehicle (TRAIN)")
    ax.axis("equal")
    ax.grid(alpha=.25)
    ax.legend(loc="best",fontsize=8)
    return str(_save(fig,Path(images_dir)/"trajectory_tracking.png"))


def plot_frozen_comparison(comparison, images_dir):
    df=pd.DataFrame(comparison)
    trained=df[df["controller"].str.startswith("trained_episode_")].copy()
    if trained.empty:
        raise ValueError("No learned frozen checkpoints")
    trained["epoch"]=trained["controller"].str.rsplit("_",n=1).str[-1].astype(int)
    trained=trained.sort_values("epoch")
    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(trained["epoch"],trained["mean_cte_post_m"]*100.,
            marker="o",label="Frozen Actor–Critic")
    for baseline,label in (
        ("untrained_deterministic","Untrained Actor"),
        ("strong_fixed_baseline","Strong fixed NMPC"),
        ("stage_lower_terminal_upper","Strong fixed alternative"),
    ):
        subset=df[df["controller"]==baseline]
        if len(subset)==1:
            ax.axhline(float(subset["mean_cte_post_m"].iloc[0])*100.,
                       linestyle="--",label=label)
    ax.set(xlabel="Completed TRAIN episodes",
           ylabel="Mean CTE after 1.3m (cm)",
           title="Frozen deterministic tracking vs fixed NMPC")
    ax.grid(alpha=.25)
    ax.legend(loc="best",fontsize=8)
    return str(_save(fig,Path(images_dir)/"frozen_tracking_comparison.png"))
