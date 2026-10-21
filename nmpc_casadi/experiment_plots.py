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
    learned=sorted(label for label in labels
                   if label.startswith("trained_episode_"))
    # Show initial, middle and final checkpoint for ANY length: 10, 100...
    if len(learned)>3:
        learned=[learned[0],learned[len(learned)//2],learned[-1]]
    preferred=[
        "untrained_deterministic",
        *learned,
        "strong_fixed_baseline",
        "stage_lower_terminal_upper",
    ]
    selected=[x for x in preferred if x in traces] or labels
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


def plot_best_train_episode(trace, episode_1based, images_dir):
    """The best qualifying stochastic TRAIN episode (NOT a frozen replay)."""
    df=pd.DataFrame(trace)
    required=("cte_m","reward","progress_m")
    if df.empty or any(x not in df for x in required):
        raise ValueError("Missing best TRAIN episode tracking/reward data")
    xy_columns=("X_m","Y_m","X_ref_m","Y_ref_m")
    has_xy=all(column in df for column in xy_columns)
    if has_xy and not np.isfinite(df[list(xy_columns)].to_numpy(float)).all():
        raise ValueError("Non-finite best TRAIN episode XY data")
    fig,axes=plt.subplots(3 if has_xy else 2,1,
                          figsize=(10,13 if has_xy else 7))
    if has_xy:
        xy=axes[0]
        x=df["X_m"].to_numpy(float)
        y=df["Y_m"].to_numpy(float)
        if all(column in df for column in ("X_next_m","Y_next_m")):
            endpoint=df[["X_next_m","Y_next_m"]].iloc[-1].to_numpy(float)
            if not np.isfinite(endpoint).all():
                plt.close(fig)
                raise ValueError("Non-finite best TRAIN endpoint")
            x=np.r_[x,endpoint[0]]
            y=np.r_[y,endpoint[1]]
        xy.plot(df["X_ref_m"],df["Y_ref_m"],"k--",label="Reference TRAIN")
        xy.plot(x,y,label=f"Actual stochastic episode {episode_1based}")
        xy.scatter(x[0],y[0],marker="o",label="Start",zorder=3)
        xy.scatter(x[-1],y[-1],marker="x",label="End",zorder=3)
        xy.set(xlabel="X (m)",ylabel="Y (m)",
               title="Best TRAIN episode: actual vehicle trajectory")
        xy.set_aspect("equal",adjustable="datalim")
        xy.legend(loc="best")
        xy.grid(alpha=.25)
        axes=axes[1:]
    axes[0].plot(df["progress_m"],100.*df["cte_m"])
    axes[0].set(ylabel="CTE (cm)",
                title=f"Best stochastic TRAIN episode {episode_1based} — completed")
    axes[1].plot(df["progress_m"],df["reward"],
                 label="Paper Eq.(4) reward / step")
    if "training_reward" in df:
        axes[1].plot(df["progress_m"],df["training_reward"],
                     label="Training reward with safety penalties",alpha=.8)
    axes[1].set(xlabel="Path progress (m)",ylabel="Reward / step")
    axes[1].legend(loc="best")
    for ax in axes: ax.grid(alpha=.25)
    fig.tight_layout()
    return str(_save(fig,Path(images_dir)/"best_train_episode.png"))


def plot_best_frozen_episode(reference, best_trace, best_label,
                             images_dir, comparator_trace=None):
    """Separate true-vehicle XY, CTE and adaptive Q/R/Qf plots for best frozen.

    The best was selected ONLY among evaluated, successfully completed
    deterministic checkpoint rollouts. Caller passes its saved complete log.
    """
    df=pd.DataFrame(best_trace)
    ref=pd.read_csv(reference)
    required=("X_m","Y_m","cross_track_m","s_progress_m")
    if df.empty or any(x not in df for x in required):
        raise ValueError("Missing best frozen vehicle XY/CTE/progress")
    result={}
    fig,ax=plt.subplots(figsize=(10,8))
    ax.plot(ref["X_ref_m"],ref["Y_ref_m"],linestyle="--",
            lw=2,label="Reference TRAIN")
    ax.plot(df["X_m"],df["Y_m"],label=f"Best frozen {best_label}")
    if comparator_trace is not None:
        fixed=pd.DataFrame(comparator_trace)
        ax.plot(fixed["X_m"],fixed["Y_m"],
                label="Strong fixed NMPC",alpha=.8)
    ax.axis("equal")
    ax.grid(alpha=.25)
    ax.set(xlabel="X (m)",ylabel="Y (m)",
           title="Best frozen episode: real vehicle XY vs reference")
    ax.legend(loc="best")
    result["xy"]=str(_save(fig,Path(images_dir)/"best_frozen_trajectory.png"))

    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(df["s_progress_m"],df["cross_track_m"]*100,
            label=f"Best frozen {best_label}")
    if comparator_trace is not None:
        ax.plot(fixed["s_progress_m"],fixed["cross_track_m"]*100,
                label="Strong fixed NMPC",alpha=.8)
    ax.set(xlabel="Path progress (m)",ylabel="CTE (cm)",
           title="Best frozen episode: cross-track error")
    ax.grid(alpha=.25)
    ax.legend(loc="best")
    result["cte"]=str(_save(fig,Path(images_dir)/"best_frozen_cte.png"))

    weight_cols=[*[f"q_{i}" for i in range(6)],
                 *[f"r_{i}" for i in range(2)],
                 *[f"p_{i}" for i in range(6)]]
    if not all(c in df for c in weight_cols):
        raise ValueError("Best frozen log is missing Q/R/Qf histories")
    fig,axes=plt.subplots(3,1,figsize=(10,10),sharex=True)
    for ax,prefix,n,labels,ylabel in (
        (axes[0],"q",6,STATE_LABELS,"Q diagonal"),
        (axes[1],"r",2,ACTION_LABELS,"R diagonal"),
        (axes[2],"p",6,STATE_LABELS,"Qf diagonal"),
    ):
        for i in range(n):
            ax.plot(df["s_progress_m"],df[f"{prefix}_{i}"],
                    label=labels[i],lw=1.15)
        ax.set(ylabel=ylabel)
        ax.grid(alpha=.25)
        ax.legend(loc="best",fontsize=8,ncol=3)
    axes[-1].set_xlabel("Path progress (m)")
    axes[0].set_title("Best frozen episode: adaptive NMPC cost weights")
    fig.tight_layout()
    result["weights"]=str(_save(fig,Path(images_dir)/"best_frozen_weights.png"))
    return result
