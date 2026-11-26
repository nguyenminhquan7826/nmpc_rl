"""Offline analysis of native hardware_runner logs. Never opens serial."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

MODES = ("fixed", "actor_adapt", "critic_adapt", "full_adapt")
COLORS = dict(zip(MODES, ("#1f77b4", "#e67e22", "#2ca02c", "#9467bd")))


def read_summary(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def finite_values(values):
    v = np.asarray(values, dtype=float)
    return v[np.isfinite(v)]


def stats(values):
    v = finite_values(values)
    if not len(v):
        return dict(count=0, mean=None, rmse=None, p95=None, max=None)
    return dict(count=int(len(v)), mean=float(v.mean()),
                rmse=float(np.sqrt(np.mean(v*v))),
                p95=float(np.percentile(v, 95)), max=float(v.max()))


def numbers(frame, name):
    if name not in frame:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float)


def value(summary, name):
    x = summary.get(name)
    if x is None:
        return None
    n = float(x)
    return n if np.isfinite(n) else None


def active_bounds(summary, events):
    runs = events[events.event == "tx_RUN"]
    stops = events[events.event == "tx_STOP"]
    if runs.empty:
        return None, None
    start = float(runs.host_s.iloc[0])
    after = stops[stops.host_s >= start]
    end = float(after.host_s.iloc[0]) if len(after) else None
    return start, end


def read_run(folder, condition):
    s = read_summary(folder / "summary.txt")
    mode = s["mode"]
    if mode not in MODES:
        raise ValueError(f"{folder}: unknown mode {mode}")
    c = pd.read_csv(folder / "control.csv")
    tr = pd.read_csv(folder / "transitions.csv")
    ev = pd.read_csv(folder / "events.csv")
    ref = pd.read_csv(folder / "reference.csv")
    required = {"host_s", "step", "mode", "X", "Y", "yaw", "vx", "progress",
                "cte", "solver_success", "clipped", "solve_ms", "reference0",
                "reference2", "reference4", "reference5"}
    if not required.issubset(c.columns):
        raise ValueError(f"{folder}: missing control columns {required-set(c.columns)}")
    if not {"s_m", "X_ref_m", "Y_ref_m"}.issubset(ref.columns) or len(ref) < 2:
        raise ValueError(f"{folder}: invalid reference")
    if len(c) and not c["mode"].eq(mode).all():
        raise ValueError(f"{folder}: inconsistent control mode")
    if not np.isfinite(c[list(required-{"mode"})].to_numpy(dtype=float)).all():
        raise ValueError(f"{folder}: nonfinite required control data")
    if not np.isfinite(ref[["s_m","X_ref_m","Y_ref_m"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{folder}: nonfinite reference")
    if not (np.diff(ref.s_m) > 0).all():
        raise ValueError(f"{folder}: nonmonotonic reference")
    if len(c) and ((np.diff(c.host_s) < 0).any() or (np.diff(c.step) <= 0).any()):
        raise ValueError(f"{folder}: control rows out of order")
    if len(c) and (c.cte < 0).any():
        raise ValueError(f"{folder}: negative logged CTE")
    warnings = []
    drive = s.get("drive") == "1"
    if not drive:
        warnings.append("Shadow/replay: no actuation or online learning inferred.")
    if condition != "vehicle":
        warnings.append("Not designated a moving-vehicle experiment; tracking performance not established.")
    warnings.append("Tracking metrics use controller odometry, not independent ground truth.")
    if s.get("final_stop_confirmed") != "1" and drive:
        warnings.append("Final STOP ACK unavailable or unconfirmed.")
    if s.get("reason") not in ("duration_completed", "route_completed", "completed"):
        warnings.append("Run did not complete normally; retain in report.")
    if value(s, "control_steps") != len(c):
        warnings.append("Control row count differs from summary.")
    if drive and value(s, "feedback_count") != len(tr):
        warnings.append("Transition row count differs from summary.")
    start, end = active_bounds(s, ev)
    duration = value(s, "active_duration_s")
    duration_source = "summary"
    if duration is None:
        duration = end-start if start is not None and end is not None else None
        duration_source = "first_RUN_to_next_STOP" if duration is not None else "unavailable"
    # Relative plot time zero is first logged control slot (proposal before RUN).
    # Preserve original host_s in the source files and in derived CSV.
    t0 = float(c.slot_host_s.iloc[0] if "slot_host_s" in c else c.host_s.iloc[0]) if len(c) else 0
    c = c.copy()
    c["relative_s"] = numbers(c, "slot_host_s" if "slot_host_s" in c else "host_s") - t0
    heading = numbers(c,"yaw") - numbers(c,"reference2")
    c["heading_error_deg"] = np.rad2deg(np.arctan2(np.sin(heading), np.cos(heading)))
    c["speed_error_mps"] = numbers(c,"vx") - numbers(c,"reference0")
    c["position_error_m"] = np.hypot(numbers(c,"X")-numbers(c,"reference4"),
                                      numbers(c,"Y")-numbers(c,"reference5"))
    tr = tr.copy()
    tr["relative_s"] = numbers(tr,"host_s") - t0
    length = float(ref.s_m.iloc[-1] - ref.s_m.iloc[0])
    progress = finite_values(c.progress)
    rewards = finite_values(numbers(tr,"reward"))
    # A terminal route-completed proposal can have no actual solve.
    # Do not treat all unsuccessful proposal rows as solver failures.
    solved = c[c.solver_success == 1]
    actor_rows = tr[numbers(tr,"batch_size") > 0]
    metrics = {
        "run": folder.name, "path": str(folder.resolve()), "mode": mode,
        "condition": condition, "drive": drive, "reason": s.get("reason"),
        "route_completed": s.get("reason") == "route_completed",
        "final_stop_confirmed": s.get("final_stop_confirmed"),
        "seed": value(s,"seed"), "control_rows": len(c), "transition_rows": len(tr),
        "active_duration_s": duration, "duration_source": duration_source,
        "route_length_m": length,
        "last_logged_progress_m": float(progress[-1]) if len(progress) else None,
        "max_logged_progress_percent": float(np.clip((progress.max()-ref.s_m.iloc[0])/length*100,0,100)) if len(progress) else None,
        "cte_m": stats(c.cte),
        "heading_abs_error_deg": stats(np.abs(c.heading_error_deg)),
        "speed_abs_error_mps": stats(np.abs(c.speed_error_mps)),
        "position_error_m": stats(c.position_error_m),
        "total_reward": float(rewards.sum()) if len(rewards) else None,
        "mean_reward": float(rewards.mean()) if len(rewards) else None,
        "td": stats(numbers(tr,"td")),
        "successful_solve_ms": stats(numbers(solved,"solve_ms")),
        "control_compute_ms": stats(numbers(c,"control_compute_ms")),
        "host_period_ms": stats(numbers(c,"period_ms")),
        "mcu_feedback_interval_ms": stats(numbers(c,"feedback_interval_ms")),
        "sample_receipt_age_ms": stats(numbers(c,"sample_receipt_age_ms")),
        "run_ack_wait_ms": stats(numbers(tr,"run_ack_wait_ms")),
        "actor_update_ms": stats(numbers(actor_rows,"actor_update_ms")),
        "critic_update_ms": stats(numbers(tr,"critic_update_ms")) if mode in ("critic_adapt","full_adapt") and drive else stats([]),
        "actor_updates": value(s,"actor_updates"), "critic_updates": value(s,"critic_updates"),
        "drive_deadline_misses": value(s,"drive_deadline_misses"),
        "successful_proposals": len(solved),
        "clipped_proposals": int((c.clipped == 1).sum()),
        "final_stop_ack_wait_ms": value(s,"final_stop_ack_wait_ms"),
        "initial_X_m": float(c.X.iloc[0]) if len(c) else None,
        "initial_Y_m": float(c.Y.iloc[0]) if len(c) else None,
        "initial_yaw_rad": float(c.yaw.iloc[0]) if len(c) else None,
        "odometry_displacement_m": float(np.hypot(c.X.iloc[-1]-c.X.iloc[0],c.Y.iloc[-1]-c.Y.iloc[0])) if len(c) else None,
        "reference_sha256": digest(folder/"reference.csv"),
        "settings_sha256": digest(folder/"settings.txt"),
        "initial_model_sha256": digest(folder/"initial.model"),
        "warnings": warnings,
    }
    return dict(metrics=metrics, control=c, transitions=tr, reference=ref, summary=s)


def flatten(metrics):
    row = {}
    for key, val in metrics.items():
        if isinstance(val, dict):
            row.update({f"{key}_{k}": v for k,v in val.items()})
        elif key == "warnings":
            row[key] = " | ".join(val)
        else:
            row[key] = val
    return row


def plot_run(run, dest, plt):
    c, tr, ref, m = (run[k] for k in ("control","transitions","reference","metrics"))
    if c.empty:
        return
    t = c.relative_s
    fig, ax = plt.subplots(2,2,figsize=(10,7), constrained_layout=True)
    ax[0,0].plot(ref.X_ref_m,ref.Y_ref_m,"k--",label="Reference")
    ax[0,0].plot(c.X,c.Y,label="Odometry")
    ax[0,0].set(xlabel="X (m)",ylabel="Y (m)",aspect="equal")
    ax[0,0].legend()
    ax[0,1].plot(t,c.cte*100)
    ax[0,1].set(ylabel="Logged CTE (cm)",xlabel="Time from first control slot (s)")
    ax[1,0].plot(t,c.vx,label="Encoder speed")
    ax[1,0].plot(t,c.reference0,"k--",label="Path reference")
    ok = c.solver_success == 1
    if "speed_request" in c:
        ax[1,0].plot(t[ok],c.loc[ok,"speed_request"],label="Proposed setpoint",alpha=.75)
    ax[1,0].set(ylabel="Speed (m/s)",xlabel="Time (s)")
    ax[1,0].legend(fontsize=8)
    for k,label in (("delta_nmpc","NMPC proposal"),("delta_quantized","Quantized proposal")):
        if k in c:
            ax[1,1].plot(t[ok],np.rad2deg(c.loc[ok,k]),label=label)
    ax[1,1].set(xlabel="Time (s)",ylabel="Steering request (deg)")
    if ax[1,1].lines:
        ax[1,1].legend(fontsize=8)
    fig.suptitle(f"{m['mode']} | {m['condition']} | odometry, not ground truth")
    for a in ax.flat:
        a.grid(alpha=.2)
    fig.savefig(dest/"tracking.png",dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
    for a, prefix, n in zip(ax,("Q","R","P"),(6,2,6)):
        for i in range(n):
            k = f"{prefix}{i}"
            if k in c:
                a.plot(t,c[k],label=k)
        a.set_ylabel(prefix)
        if a.lines:
            a.legend(ncol=3,fontsize=8)
        a.grid(alpha=.2)
    ax[-1].set_xlabel("Time (s)")
    fig.suptitle("Weights used in logged proposals")
    fig.savefig(dest/"weights.png",dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
    for k,label in (("solve_ms","Solver"),("control_compute_ms","Total computation")):
        if k in c:
            ax[0].plot(t[ok],c.loc[ok,k],label=label)
    if "run_ack_wait_ms" in tr:
        ax[0].plot(tr.relative_s,tr.run_ack_wait_ms,label="RUN ACK wait",alpha=.7)
    ax[0].set_ylabel("Time (ms)")
    if ax[0].lines: ax[0].legend(fontsize=8)
    for k,label in (("period_ms","Host period"),("feedback_interval_ms","MCU feedback interval")):
        if k in c: ax[1].plot(t,c[k],label=label)
    if ax[1].lines: ax[1].legend(fontsize=8)
    ax[1].set_ylabel("Interval (ms)")
    if "sample_receipt_age_ms" in c: ax[2].plot(t,c.sample_receipt_age_ms)
    ax[2].set(ylabel="Host receipt age (ms)",xlabel="Time (s)")
    for a in ax: a.grid(alpha=.2)
    fig.savefig(dest/"timing.png",dpi=180)
    plt.close(fig)

    if len(tr):
        fig, ax = plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
        ax[0].plot(tr.relative_s,tr.reward)
        ax[0].set_ylabel("Reward")
        ax[1].plot(tr.relative_s,tr.td)
        ax[1].set_ylabel("TD error (as logged)")
        for k in ("actor_step_norm","critic_step_norm"):
            if k in tr: ax[2].plot(tr.relative_s,tr[k],label=k)
        if ax[2].lines: ax[2].legend(fontsize=8)
        ax[2].set(xlabel="Time (s)",ylabel="Update norm")
        for a in ax: a.grid(alpha=.2)
        fig.savefig(dest/"learning.png",dpi=180)
        plt.close(fig)


def comparison(runs, out, plt):
    metrics = [r["metrics"] for r in runs]
    checks = {}
    for k in ("reference_sha256","settings_sha256","initial_model_sha256"):
        vals = [m[k] for m in metrics]
        checks[k+"_match"] = all(v is not None and v == vals[0] for v in vals)
    checks["all_drive"] = all(m["drive"] for m in metrics)
    checks["all_designated_vehicle"] = all(m["condition"] == "vehicle" for m in metrics)
    checks["all_stop_confirmed"] = all(m["final_stop_confirmed"] == "1" for m in metrics)
    checks["all_normal_completion"] = all(m["reason"] in ("duration_completed","route_completed") for m in metrics)
    checks["same_seed"] = all(m["seed"] is not None and m["seed"] == metrics[0]["seed"] for m in metrics)
    checks["initial_pose_verified"] = False  # logged pose is not independent evidence
    checks["ground_truth_available"] = False
    checks["note"] = ("Matching files do not establish equal physical initial conditions or duration. "
                      "Different run lengths affect total reward. No ranking or ground-truth claim.")
    (out/"comparison_checks.json").write_text(json.dumps(checks,indent=2)+"\n")
    if plt is None or len(runs)<2:
        return
    fig, ax = plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    if checks["reference_sha256_match"]:
        ref = runs[0]["reference"]
        ax[0,0].plot(ref.X_ref_m,ref.Y_ref_m,"k--",label="Reference")
    for i,run in enumerate(runs,1):
        c,m = run["control"],run["metrics"]
        if c.empty: continue
        label = f"{m['mode']} ({i})"
        color = COLORS[m["mode"]]
        ax[0,0].plot(c.X,c.Y,label=label,color=color)
        ax[0,1].plot(c.relative_s,c.cte*100,label=label,color=color)
        ax[1,0].plot(c.relative_s,c.vx,label=label,color=color)
    ax[0,0].set(xlabel="X (m)",ylabel="Y (m)",aspect="equal")
    ax[0,1].set(xlabel="Time (s)",ylabel="Logged CTE (cm)")
    ax[1,0].set(xlabel="Time (s)",ylabel="Encoder speed (m/s)")
    labels = [f"{m['mode']}\n{i}" for i,m in enumerate(metrics,1)]
    x = np.arange(len(metrics))
    mean = [m["control_compute_ms"]["mean"] for m in metrics]
    worst = [m["control_compute_ms"]["max"] for m in metrics]
    ax[1,1].bar(x-.18,[np.nan if v is None else v for v in mean],.36,label="Mean")
    ax[1,1].bar(x+.18,[np.nan if v is None else v for v in worst],.36,label="Max")
    ax[1,1].set_xticks(x,labels,rotation=20)
    ax[1,1].set_ylabel("Control computation (ms)")
    for a in ax.flat:
        a.grid(alpha=.2)
        if a.get_legend_handles_labels()[0]: a.legend(fontsize=8)
    fig.suptitle("Run comparison | odometry only; review comparison_checks.json")
    fig.savefig(out/"comparison.png",dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs",nargs="*",type=Path,help="Explicit run directories (repeat trials allowed)")
    p.add_argument("--latest",nargs="+",choices=(*MODES,"all"),help="Select newest drive run for each requested mode")
    p.add_argument("--root",type=Path,default=Path("hardware_runs"))
    p.add_argument("--condition",choices=("unknown","bench","vehicle"),default="unknown",
                   help="Operator-supplied condition for all selected runs")
    p.add_argument("--output",type=Path)
    p.add_argument("--no-plots",action="store_true")
    a = p.parse_args()
    paths = list(a.runs)
    if a.latest:
        modes = MODES if "all" in a.latest else tuple(dict.fromkeys(a.latest))
        for mode in modes:
            candidates = [d for d in a.root.glob(f"drive_{mode}_*") if d.is_dir()]
            if not candidates: p.error(f"No drive directory for {mode}")
            # Runner names end in epoch microseconds and pid. Include failed
            # latest trials rather than silently falling back to older success.
            candidates.sort(key=lambda d:d.name)
            paths.append(candidates[-1])
    if not paths: p.error("Provide run directory or --latest fixed|...|all")
    paths = list(dict.fromkeys(d.resolve() for d in paths))
    runs = [read_run(d,a.condition) for d in paths]
    output = a.output or Path("hardware_analysis")/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True,exist_ok=False)
    plt = None
    if not a.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    for i,run in enumerate(runs,1):
        dest = output/f"{i:02d}_{run['metrics']['run']}"
        dest.mkdir()
        run["control"].to_csv(dest/"derived_control.csv",index=False)
        (dest/"metrics.json").write_text(json.dumps(run["metrics"],indent=2,allow_nan=False)+"\n")
        if plt is not None: plot_run(run,dest,plt)
    pd.DataFrame([flatten(r["metrics"]) for r in runs]).to_csv(output/"summary.csv",index=False)
    comparison(runs,output,plt)
    print(f"Output: {output}")
    print(pd.DataFrame([flatten(r["metrics"]) for r in runs])[
        ["mode","condition","reason","control_rows","active_duration_s","cte_m_rmse",
         "mean_reward","control_compute_ms_max","actor_updates","critic_updates"]].to_string(index=False))
    print("Tracking statistics are based on odometry; not independent ground truth.")


if __name__ == "__main__":
    main()
