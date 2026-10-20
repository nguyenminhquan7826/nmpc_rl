"""Plot Direct-P training and held-out evaluation logs without changing the controller.

PNG plots are generated automatically by train_direct_p.py and
eval_direct_p_heldout.py. Re-plot previously saved CSV logs without retraining:

    python plot_direct_p_logs.py --train-dir direct_results/train_direct_p_ep100_roll16 \
        --eval-dir direct_results/eval_direct_p_ep100_roll16 --plot-every 10

Only existing CSV data are read. No Actor/Critic parameters are changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive/CI/Windows-compatible image output.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


P_NAMES = ("v_x", "v_y", "psi", "r", "X", "Y")
CONTROLLER_NAMES = {
    "fixed_nmpc": "Fixed NMPC",
    "actor_frozen_initial_p_sampled": "Actor NN + initial P",
    "direct_p_frozen_sampled": "Actor NN + learned P, frozen",
    "direct_p_frozen_mean": "Actor NN + learned P, mean",
    "direct_p_online_sampled": "Actor NN + learned P, online",
}


def _save(fig, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return target


def _line(ax, data, x, y, label=None, factor=1.0):
    if x not in data or y not in data:
        return False
    xval = pd.to_numeric(data[x], errors="coerce").to_numpy(float)
    yval = pd.to_numeric(data[y], errors="coerce").to_numpy(float) * factor
    mask = np.isfinite(xval) & np.isfinite(yval)
    if not np.any(mask):
        return False
    ax.plot(xval[mask], yval[mask], label=label if label else y, linewidth=1.35)
    return True


def _finish_axis(ax, title, xlabel, ylabel, legend=False, **kwargs):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28)
    if legend and ax.lines:
        ax.legend(fontsize=8, **kwargs)


def plot_training_history(history, output):
    """Plot reward, tracking, terminal P evolution and solver statistics."""
    history = pd.DataFrame(history)
    if history.empty or "episode" not in history:
        return
    output = Path(output)
    specs = (
        ("return", "training_return.png", "Training return", "Return", 1.0),
        ("mean_cte_post_m", "training_mean_cte_post.png",
         "Training mean CTE after s = 1.3 m", "Mean CTE (cm)", 100.0),
        ("rmse_heading_post_rad", "training_heading_post.png",
         "Training heading RMSE after s = 1.3 m", "Heading RMSE (rad)", 1.0),
        ("solver_success_pct", "training_solver_success.png",
         "Training NMPC solver success", "Solver success (%)", 1.0),
    )
    for column, filename, title, ylabel, factor in specs:
        if column not in history:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.5))
        _line(ax, history, "episode", column, factor=factor)
        _finish_axis(ax, title, "Episode", ylabel)
        _save(fig, output / filename)

    p_cols = [f"p_final_{i}" for i in range(6)]
    if all(c in history for c in p_cols):
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for i, col in enumerate(p_cols):
            _line(ax0, history, "episode", col, f"P_{P_NAMES[i]}")
            init = (240., 60., 1000., 120., 2015., 2015.)[i]
            _line(ax1, history.assign(**{f"ratio_{i}": history[col] / init}),
                  "episode", f"ratio_{i}", f"P_{P_NAMES[i]} / initial")
        _finish_axis(ax0, "Learned terminal Qf (physical diagonal)",
                     "", "Qf weight", legend=True, ncol=3)
        _finish_axis(ax1, "Change from nominal terminal diagonal",
                     "Episode", "Multiplier", legend=True, ncol=3)
        _save(fig, output / "training_terminal_weights.png")


def plot_trajectory(logs, target, title, reference=None):
    """Plot XY closed-loop trajectory and the complete reference where available."""
    fig, ax = plt.subplots(figsize=(8, 7))
    if reference is not None and {"X_ref_m", "Y_ref_m"} <= set(reference.columns):
        ax.plot(reference["X_ref_m"], reference["Y_ref_m"],
                "k--", linewidth=2, label="Reference (full route)")
    elif {"X_ref", "Y_ref"} <= set(logs.columns):
        ax.plot(logs["X_ref"], logs["Y_ref"],
                "k--", linewidth=2, label="Reference")
    ax.plot(logs["X"], logs["Y"], linewidth=1.4, label="Vehicle")
    _finish_axis(ax, title, "X (m)", "Y (m)", legend=True)
    ax.set_aspect("equal", adjustable="datalim")
    return _save(fig, target)


def plot_weights(logs, target, title):
    """Every diagonal Q(6), R(2), Qf(6) over physical runtime, not update count."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, prefix, length, ylabel in (
        (axes[0], "q", 6, "Stage Q"),
        (axes[1], "r", 2, "Input R"),
        (axes[2], "qf", 6, "Terminal Qf / P"),
    ):
        for i in range(length):
            col = f"{prefix}_{i}"
            label = f"{prefix.upper()}_{P_NAMES[i]}" if length == 6 else f"R_{i}"
            _line(ax, logs, "t_s", col, label)
        _finish_axis(ax, ylabel, "", ylabel, legend=True, ncol=3)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Time (s)")
    return _save(fig, target)


def plot_cte(logs, target, title):
    xcol = "progress_m" if "progress_m" in logs else "t_s"
    fig, ax = plt.subplots(figsize=(10, 4.6))
    _line(ax, logs, xcol, "cte_m", "CTE", factor=100.)
    _finish_axis(ax, title, "Progress (m)" if xcol == "progress_m" else "Time (s)",
                 "Cross-track error (cm)")
    return _save(fig, target)


def plot_updates(updates, target, title):
    """TD RMS and Actor/Critic losses, preserving separate numerical scales."""
    if updates is None or updates.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    _line(axes[0], updates, "step", "td_rms", "TD RMS")
    _line(axes[0], updates, "step", "td_mean", "TD mean")
    _finish_axis(axes[0], title, "", "TD error", legend=True)
    _line(axes[1], updates, "step", "critic_loss_total", "Critic loss")
    _finish_axis(axes[1], "Critic TD(0) loss", "", "Critic loss", legend=True)
    _line(axes[2], updates, "step", "actor_loss_total", "Actor total loss")
    _finish_axis(axes[2], "Actor policy-gradient loss", "Step", "Actor loss", legend=True)
    return _save(fig, target)


def plot_training_episode(output, episode, logs, updates=None):
    """Generate selected episode plots; call at --plot-every intervals."""
    root = Path(output) / "plots"
    name = f"episode_{int(episode):04d}"
    plot_trajectory(logs, root / f"{name}_trajectory.png",
                    f"TRAIN episode {episode}: XY tracking")
    plot_cte(logs, root / f"{name}_cte_response.png",
             f"TRAIN episode {episode}: cross-track error")
    plot_weights(logs, root / f"{name}_weights_qrqf.png",
                 f"TRAIN episode {episode}: Q / R / Qf")
    if updates is not None and not updates.empty:
        plot_updates(updates, root / f"{name}_td_losses.png",
                     f"TRAIN episode {episode}: TD diagnostics")


def _full_reference(output):
    output = Path(output)
    names = [output / "cad_eval_source_xref.csv"]
    names.extend(sorted(output.glob("*_reference.csv")))
    for path in names:
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path)
            if {"X_ref_m", "Y_ref_m"} <= set(df.columns):
                return df
        except (OSError, pd.errors.ParserError):
            continue
    return None


def plot_profile_overlay(profile, logs, output, reference=None):
    """Profile-wide XY, CTE, heading and actuator comparison."""
    output = Path(output)
    if not logs:
        return
    fig, ax = plt.subplots(figsize=(8.5, 7))
    if reference is not None:
        ax.plot(reference["X_ref_m"], reference["Y_ref_m"],
                "k--", linewidth=2, label="Reference (full)")
    else:
        first = next(iter(logs.values()))
        ax.plot(first["X_ref"], first["Y_ref"],
                "k--", linewidth=2, label="Reference")
    for key, df in logs.items():
        ax.plot(df["X"], df["Y"], linewidth=1.2,
                label=CONTROLLER_NAMES.get(key, key))
    _finish_axis(ax, f"{profile}: tracking trajectory", "X (m)", "Y (m)",
                 legend=True, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    _save(fig, output / f"{profile}_trajectory.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for key, df in logs.items():
        _line(ax, df, "progress_m", "cte_m",
              CONTROLLER_NAMES.get(key, key), factor=100.)
    _finish_axis(ax, f"{profile}: cross-track error vs progress", "Progress (m)",
                 "CTE (cm)", legend=True)
    _save(fig, output / f"{profile}_fixed_vs_ac_response.png")

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for key, df in logs.items():
        label = CONTROLLER_NAMES.get(key, key)
        _line(axes[0], df, "progress_m", "cte_m", label, factor=100.)
        _line(axes[1], df, "progress_m", "e_psi", label, factor=180. / np.pi)
        _line(axes[2], df, "progress_m", "delta", label, factor=180. / np.pi)
    _finish_axis(axes[0], f"{profile}: Fixed NMPC vs Actor/Critic", "",
                 "CTE (cm)", legend=True, ncol=2)
    _finish_axis(axes[1], "Heading error", "", "Heading error (deg)")
    _finish_axis(axes[2], "Steering angle", "Progress (m)", "Steering (deg)")
    _save(fig, output / f"{profile}_fixed_vs_ac.png")


def plot_metric_comparison(table, output):
    if table is None or table.empty or "profile" not in table:
        return
    output = Path(output)
    metrics = (
        ("mean_cte_post_m", "Mean CTE post (cm)", 100.),
        ("max_cte_post_m", "Max CTE post (cm)", 100.),
        ("rmse_heading_post_rad", "Heading RMSE post (rad)", 1.),
        ("solver_success_pct", "Solver success (%)", 1.),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (col, ylabel, multiplier) in zip(axes.flat, metrics):
        for controller, df in table.groupby("controller", sort=False):
            if col in df.columns:
                ax.plot(df["profile"], pd.to_numeric(df[col], errors="coerce") *
                        multiplier, marker="o", label=CONTROLLER_NAMES.get(
                            controller, controller))
        _finish_axis(ax, ylabel, "Plant profile", ylabel)
    axes[0, 0].legend(fontsize=8, ncol=2)
    _save(fig, output / "tracking_metric_comparison.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for controller, df in table.groupby("controller", sort=False):
        if "core_loop_p95_ms" in df.columns:
            ax.plot(df["profile"], df["core_loop_p95_ms"], marker="o",
                    label=CONTROLLER_NAMES.get(controller, controller))
    _finish_axis(ax, "Core controller loop p95 time", "Plant profile",
                 "p95 (ms)", legend=True)
    _save(fig, output / "timing_fixed_vs_ac.png")


def plot_evaluation_output(output):
    """Reconstruct all plots from previously saved held-out CSV logs."""
    output = Path(output)
    table = pd.read_csv(output / "comparison.csv")
    plot_metric_comparison(table, output)
    reference = _full_reference(output)
    for profile, subset in table.groupby("profile", sort=False):
        logs = {}
        for controller in subset["controller"]:
            path = output / f"{profile}_{controller}.csv"
            if not path.exists():
                continue
            log = pd.read_csv(path)
            logs[controller] = log
            plot_weights(log, output / f"{profile}_{controller}_weights_qrqf.png",
                         f"TEST {profile}: {CONTROLLER_NAMES.get(controller, controller)}")
            # Retain light-weight per-controller response plots too.
            plot_cte(log, output / f"{profile}_{controller}_cte_response.png",
                     f"TEST {profile}: {controller} CTE")
        plot_profile_overlay(profile, logs, output, reference)
        online = output / f"{profile}_direct_p_online_sampled_updates.csv"
        if online.exists():
            plot_updates(pd.read_csv(online),
                         output / f"{profile}_online_td_losses.png",
                         f"TEST {profile}: online TD diagnostics")


def plot_training_output(output, every=10):
    """Regenerate plots from existing training CSV files without retraining."""
    output = Path(output)
    if every < 1:
        raise ValueError("plot-every must be >= 1")
    history = pd.read_csv(output / "training_history.csv")
    plot_training_history(history, output)
    if history.empty:
        return
    episodes = set(int(x) for x in history["episode"])
    last = max(episodes)
    for episode in sorted(episodes):
        if episode != 0 and (episode + 1) % every and episode != last:
            continue
        stem = f"episode_{episode:04d}"
        log_path = output / f"{stem}.csv"
        if not log_path.exists():
            continue
        update_path = output / f"{stem}_updates.csv"
        updates = pd.read_csv(update_path) if update_path.exists() else None
        plot_training_episode(output, episode, pd.read_csv(log_path), updates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--eval-dir", type=Path)
    parser.add_argument("--plot-every", type=int, default=10)
    args = parser.parse_args()
    if args.train_dir is None and args.eval_dir is None:
        parser.error("Provide --train-dir, --eval-dir, or both")
    if args.train_dir is not None:
        plot_training_output(args.train_dir, args.plot_every)
        print(f"Training plots saved: {args.train_dir}", flush=True)
    if args.eval_dir is not None:
        plot_evaluation_output(args.eval_dir)
        print(f"Evaluation plots saved: {args.eval_dir}", flush=True)


if __name__ == "__main__":
    main()
