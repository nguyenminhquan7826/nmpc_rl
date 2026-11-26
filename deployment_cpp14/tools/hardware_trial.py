"""Prepare a common hardware input set and launch exactly one C++ controller."""
import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import subprocess
import sys

DEPLOY = Path(__file__).resolve().parents[1]
ROOT = DEPLOY.parent
MODES = ("fixed", "actor_adapt", "critic_adapt", "full_adapt")
INPUTS = DEPLOY / "hardware_trial_inputs"
CHECKPOINT = DEPLOY / "results_v11_best/checkpoint_best_frozen.npz"
CONFIG = ROOT / "nmpc_casadi/evaluation_four_modes_5/evaluation_config.json"
REFERENCE = DEPLOY / "examples/reference_straight_2m.csv"
HARDWARE = DEPLOY / "examples/hardware_nominal.json"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare", help="Export once; never overwrite inputs")
    prep.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    prep.add_argument("--config", type=Path, default=CONFIG)
    prep.add_argument("--reference", type=Path, default=REFERENCE)
    prep.add_argument("--hardware", type=Path, default=HARDWARE)
    prep.add_argument("--output", type=Path, default=INPUTS)
    prep.add_argument("--start", choices=("initial_heads", "trained"),
                      default="initial_heads")
    prep.add_argument("--horizon", type=int, default=20)
    run = sub.add_parser("run", help="One mode; RX-only shadow unless --drive")
    run.add_argument("mode", choices=MODES)
    run.add_argument("--inputs", type=Path, default=INPUTS)
    run.add_argument("--binary", type=Path, default=DEPLOY / "build/hardware_runner")
    source = run.add_mutually_exclusive_group()
    source.add_argument("--device", default=None)
    source.add_argument("--input", type=Path, help="Replay raw TEL log (shadow only)")
    run.add_argument("--duration", type=float, default=60)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--pose", type=float, nargs=3, metavar=("X_M", "Y_M", "YAW_DEG"),
                     help="Actual initial pose in reference coordinates; mandatory for drive")
    run.add_argument("--drive", action="store_true",
                     help="Explicitly enable drive with keyboard r/q interlock")
    run.add_argument("--output", type=Path, help="New result directory")
    return p


def run_command(a, p):
    if not math.isfinite(a.duration) or not 0 < a.duration <= 3600:
        p.error("duration must be in (0, 3600]")
    if not 0 <= a.seed <= 2147483647:
        p.error("seed must be in [0, 2147483647]")
    if a.pose is not None and not all(math.isfinite(v) for v in a.pose):
        p.error("pose must be finite")
    if a.drive and (a.input is not None or a.pose is None):
        p.error("drive requires live serial and explicit --pose X_M Y_M YAW_DEG")
    binary = a.binary.resolve()
    inputs = a.inputs.resolve()
    model, settings, reference = (inputs / n for n in
                                  ("initial.model", "hardware.settings", "reference.csv"))
    for f in (binary, model, settings, reference):
        if not f.is_file():
            p.error(f"Missing {f}; build or run prepare first")
    if not os.access(binary, os.X_OK):
        p.error(f"Not executable: {binary}")
    if a.input is not None and not a.input.is_file():
        p.error(f"Missing replay log: {a.input}")
    kind = "drive" if a.drive else ("replay" if a.input else "shadow")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (a.output or DEPLOY / "hardware_runs" /
              f"{kind}_{a.mode}_{stamp}").resolve()
    if output.exists():
        p.error(f"Output already exists: {output}")
    cmd = [str(binary), "--model", str(model), "--settings", str(settings),
           "--reference", str(reference), "--mode", a.mode,
           "--duration", str(a.duration), "--seed", str(a.seed),
           "--output", str(output)]
    if a.input is None:
        cmd.extend(["--device", a.device or "/dev/ttyACM0"])
    else:
        cmd.extend(["--input", str(a.input.resolve())])
    if a.pose is not None:
        for key, value in zip(("--x0", "--y0", "--yaw0-deg"), a.pose):
            cmd.extend([key, str(value)])
    if a.drive:
        cmd.extend(["--drive", "--keyboard"])
    return cmd, output


def main():
    p = parser()
    a = p.parse_args()
    if a.action == "prepare":
        cmd = [sys.executable, str(DEPLOY / "tools/prepare_hardware_trial.py"),
               "--checkpoint", str(a.checkpoint), "--config", str(a.config),
               "--reference", str(a.reference), "--hardware", str(a.hardware),
               "--output", str(a.output), "--start", a.start,
               "--horizon", str(a.horizon)]
        return subprocess.call(cmd)
    if a.drive and not sys.stdin.isatty():
        p.error("drive requires an interactive terminal for r/q")
    cmd, output = run_command(a, p)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"One trial: {a.mode}; output: {output}", flush=True)
    # Replace the launcher so the native runner receives terminal signals directly.
    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())
