"""Compare four online controller modes from independent copies of one checkpoint.

Adaptation modes learn on the supplied route and are not frozen test scores.
The CLI runs Fixed, Actor adapt, Critic adapt and Full adapt only.
Legacy diagnostic helpers remain available to existing analysis/tests.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
os.environ["NMPC_RL_MINIMAL_LAYOUT"]="1"
import baseline as b
from cad_eval_reference import prepare_cad_evaluation_reference, sha256_file
from direct_p_critic import SixParameterTDCritic
from evaluate_frozen_actor_critic import frozen_rollout, load_frozen_agents
from online_adaptation import adaptive_rollout
from train_actor_critic import write_checkpoint
from train_rl_environment import TrainEnvironment
from gaussian_state_sensor import GaussianStateSensor, EXAMPLE_STD
from training_initialization import TRAIN_REFERENCE_SHA256
from run_experiment import FixedWeightActor

MODES=("fixed","actor_adapt","critic_adapt","full_adapt")
ADAPTING_MODES=("actor_adapt","critic_adapt","full_adapt","critic_adapt_sampled")


def initial_fixed_policy(config,low,high):
    q=np.asarray(config["initial_Q"],float)
    r=np.asarray(config["initial_R"],float)
    # Independent terminal configuration: never borrow the adapting P.
    qf=np.asarray(config.get("fixed_Qf",config["initial_P"]),float)
    return FixedWeightActor(q,r),SixParameterTDCritic(qf,low[2],high[2])

def load_checkpoint(path, seed):
    with np.load(path, allow_pickle=False) as z:
        config = json.loads(str(z["metadata"].item()))

    # Validate the original checkpoint BEFORE applying evaluation overrides.
    low = tuple(np.asarray(config[f"{k}_lower"], dtype=float)
                for k in ("Q", "R", "P"))
    high = tuple(np.asarray(config[f"{k}_upper"], dtype=float)
                 for k in ("Q", "R", "P"))
    _, (actor, critic), _ = load_frozen_agents(
        path, low, high, expected_gamma=config["gamma"])
    config["source_initial_weights"] = {
        k: list(config[f"initial_{k}"]) for k in ("Q", "R", "P")
    }
    config["source_weight_bounds"] = {
        f"{k}_{side}": list(config[f"{k}_{side}"])
        for k in ("Q", "R", "P") for side in ("lower", "upper")
    }

    # ===== Evaluation bounds: edit each array only here =====
    # State order: vx, vy, psi, r, X, Y.
    q_lower = np.array([1, 0.2, 1, 0.5, 10, 10], dtype=float)
    q_upper = np.array([100, 50, 250, 100, 500, 500], dtype=float)
    # Input order: acceleration, steering.
    r_lower = np.array([0.05, 0.5], dtype=float)
    r_upper = np.array([30, 800], dtype=float)
    p_lower = np.array([10, 2, 20, 5, 100, 100], dtype=float)
    p_upper = np.array([1000, 300, 2000, 500, 25000, 25000], dtype=float)
    low = (q_lower, r_lower, p_lower)
    high = (q_upper, r_upper, p_upper)

    # ===== Shared nominal initialization for all reset modes =====
    config["initial_Q"] = [12, 3, 120, 6, 153.75, 152.5]
    config["initial_R"] = [15, 100]
    config["fixed_Qf"] = (
        2.0 * np.asarray(config["initial_Q"], dtype=float)
    ).tolist()
    config["initial_P"] = config["fixed_Qf"].copy()

    # Validate once, before any simulation or model reset.
    for key, lower, upper, size in zip(("Q", "R", "P"), low, high, (6, 2, 6)):
        initial = np.asarray(config[f"initial_{key}"], dtype=float)
        if (initial.shape != (size,) or lower.shape != (size,)
            or upper.shape != (size,) or not np.isfinite(initial).all()
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower <= 0) or np.any(lower >= upper)):
            raise ValueError(f"Invalid eval weights/bounds: {key}")
        valid = ((initial > lower) & (initial < upper) if key in ("Q", "R")
                 else (initial >= lower) & (initial <= upper))
        if not np.all(valid):
            raise ValueError(
                f"initial_{key} outside eval bounds: "
                f"initial={initial}, lower={lower}, upper={upper}")
        config[f"{key}_lower"] = lower.tolist()
        config[f"{key}_upper"] = upper.tolist()
    fixed_qf = np.asarray(config["fixed_Qf"], dtype=float)
    if (fixed_qf.shape != (6,) or not np.isfinite(fixed_qf).all()
        or np.any(fixed_qf < p_lower) or np.any(fixed_qf > p_upper)):
        raise ValueError("fixed_Qf outside evaluation P bounds")

    # Synchronize live bounds ONCE; output heads are reset by the selected mode.
    actor.lower = np.r_[q_lower, r_lower].copy()
    actor.upper = np.r_[q_upper, r_upper].copy()
    critic.lower = p_lower.copy()
    critic.upper = p_upper.copy()
    actor.rng = np.random.default_rng(seed)
    config["evaluation_overrides_applied"] = True

    config["actor_lr"] = 5e-4
    config["critic_lr"] = 50000.0
    config["actor_output_lr_factors"] = [1.0, 1.0, 0.5, 0.5, 2.0, 2.0, 0.5, 0.5]
    config["critic_p_lr_factors"] = [0.5, 0.5, 0.1, 0.05, 4.0, 4.0]
    return actor, critic, config, low, high


def initialize_adaptation(actor,critic,config,low,high,start):
    """Reset nominal policy outputs/P without changing input checkpoint."""
    if start=="trained":
        if config.get("evaluation_overrides_applied",False):
            raise ValueError("Custom eval bounds require initial_heads or fresh; trained would remap learned Actor outputs")
        return actor,critic
    if start not in ("initial_heads","fresh"):
        raise ValueError("Unknown adaptation start")
    from gaussian_weight_actor import GaussianWeightActor
    initial=GaussianWeightActor(np.random.default_rng(int(config["seed"])),
        nominal_q=np.array(config["initial_Q"]),nominal_r=np.array(config["initial_R"]),
        q_lower=low[0],q_upper=high[0],r_lower=low[1],r_upper=high[1],
        sigma_lower=config["sigma_lower"],sigma_initial=config["sigma_initial"],
        sigma_upper=config["sigma_upper"])
    keys=("W_mu","b_mu","W_sigma","b_sigma") if start=="initial_heads" else tuple(actor.params)
    for key in keys:actor.params[key][...]=initial.params[key]
    critic=SixParameterTDCritic(np.array(config["initial_P"]),low[2],high[2])
    return actor,critic


def run_mode(env, actor, critic, config, mode):
    if mode in ("fixed","frozen"):
        summary,log=frozen_rollout(env,actor,critic,config["gamma"],mode)
        summary.update(actor_updates=0,critic_updates=0,action_mode="deterministic_mean")
    else:
        flags={"actor_adapt":(True,False,True),
               "critic_adapt":(False,True,False),
               "critic_adapt_sampled":(False,True,True),
               "full_adapt":(True,True,True),
               "frozen_sampled":(False,False,True)}
        if mode not in flags:raise ValueError("Unknown controller mode")
        ua,uc,sample=flags[mode]
        summary,log=adaptive_rollout(env,actor,critic,config,
            update_actor=ua,update_critic=uc,sample_actions=sample)
    log["controller"]=mode
    summary.update(mode=mode,controller=mode,
        score_role="online_adaptation" if mode in ADAPTING_MODES else "frozen_evaluation",
        loop_p95_ms=float(np.percentile(log.loop_ms,95)),
        full_loop_deadline_missed_pct=float(100*(log.loop_ms>b.DEADLINE_MS).mean()),
        mean_du_a_sq=float((log.du_a**2).mean()),
        mean_du_delta_sq=float((log.du_delta**2).mean()),
        mean_cte_all_m=float(log.cross_track_m.abs().mean()),
        max_cte_all_m=float(log.cross_track_m.abs().max()))
    return summary,log


def evaluate_final_policy(env, actor, critic, config, source_mode):
    """Reset the plant, retain final agents exactly, disable sampling/learning.

    Run in memory: never reload/reset learned output heads or P. This is
    deterministic scoring on the adaptation route, not a held-out test.
    """
    summary,log=run_mode(env,actor,critic,config,"frozen")
    log["controller"]=source_mode
    summary.update(mode=source_mode,controller=source_mode,
                   score_role="post_adaptation_deterministic",
                   actor_updates=0,critic_updates=0)
    return summary,log


def plot_comparison(traces,output,*,phase="online"):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(9,8))
    first=next(iter(traces.values()))
    ax.plot(first.X_ref_m,first.Y_ref_m,"k--",label="Reference (logged)")
    for mode,log in traces.items():
        x=np.r_[log.X_m,log.X_next_m.iloc[-1]]
        y=np.r_[log.Y_m,log.Y_next_m.iloc[-1]]
        complete=bool(log.reached_end.iloc[-1])
        ax.plot(x,y,label=mode+("" if complete else " [incomplete]"),alpha=.85)
    ax.set(xlabel="X (m)",ylabel="Y (m)",title=("Controller trajectories (fixed = initial Q/R/Qf)" if phase=="online"
                         else "Final controllers: deterministic, no learning"))
    ax.axis("equal");ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/"trajectories.png",dpi=180);plt.close(fig)
    modes=list(traces);positions=np.arange(len(modes))
    means=[100*traces[m].cross_track_m.abs().mean() for m in modes]
    maxima=[100*traces[m].cross_track_m.abs().max() for m in modes]
    labels=[m+("" if bool(traces[m].reached_end.iloc[-1]) else "\n[incomplete]") for m in modes]
    fig,ax=plt.subplots(figsize=(11,5))
    for offset,values,label in ((-.2,means,"Mean |CTE|"),(.2,maxima,"Max |CTE|")):
        bars=ax.bar(positions+offset,values,.4,label=label)
        ax.bar_label(bars,fmt="%.2f",padding=3,fontsize=8)
    ax.set_xticks(positions,labels,rotation=15)
    ax.set(ylabel="CTE (cm)",title="CTE over the complete logged rollout, including initial transient")
    ax.legend();ax.grid(axis="y",alpha=.25)
    fig.tight_layout();fig.savefig(output/"cte_mean_max.png",dpi=180);plt.close(fig)
    if "full_adapt" in traces:
        log=traces["full_adapt"]
        fig,axes=plt.subplots(3,1,figsize=(11,11),sharex=True)
        for ax,prefix,names in ((axes[0],"q",("vx","vy","psi","r","X","Y")),
            (axes[1],"r",("acceleration","steering")),
            (axes[2],"p",("vx","vy","psi","r","X","Y"))):
            for i,name in enumerate(names):ax.plot(log.s_progress_m,log[f"{prefix}_{i}"],label=name)
            ax.set_ylabel({"q":"Q diagonal","r":"R diagonal","p":"Qf applied"}[prefix])
            ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.25)
        axes[-1].set_xlabel("Path progress (m)")
        fig.suptitle("Full adapt: applied Q, R and Qf during online learning" if phase=="online"
                     else "Final Full adapt controller: deterministic Q, R and frozen Qf")
        fig.tight_layout();fig.savefig(output/"full_adapt_weights.png",dpi=180);plt.close(fig)


def plot_actor_diagnostics(log, output):
    """Separate exploration from the pre-update deterministic policy outputs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,1,figsize=(11,13),sharex=True)
    x=log.s_progress_m
    for ax,sampled,det,label in ((axes[0],"q_4","q_det_4","QX"),
            (axes[1],"q_5","q_det_5","QY"),(axes[2],"r_1","r_det_1","R steering")):
        ax.plot(x,log[sampled],alpha=.35,linewidth=.7,label="Sampled/applied")
        ax.plot(x,log[det],color="black",label="Deterministic at latent mean")
        ax.set_ylabel(label)
    axes[3].plot(x,log.td,label="Pre-update TD")
    axes[3].axhline(0,color="black",linestyle="--",linewidth=.6)
    if "actor_advantage" in log:
        axes[3].plot(x,log.actor_advantage,label="Actor advantage (assigned to sampled action)",alpha=.7)
    axes[3].set_ylabel("TD / advantage")
    axes[4].plot(x,log.actor_param_step_norm.cumsum(),label="Cumulative update path length")
    axes[4].set_ylabel("Actor update path")
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[-1].set_xlabel("Path progress (m)")
    fig.suptitle("Full adapt: exploration versus deterministic policy outputs")
    fig.tight_layout();fig.savefig(Path(output)/"actor_learning_diagnostics.png",dpi=180)
    plt.close(fig)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--reference",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--max-steps",type=int,default=2000)
    ap.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    ap.add_argument("--ey",type=float,default=.05)
    ap.add_argument("--heading-deg",type=float,default=0.)
    ap.add_argument("--sampling-control",action="store_true",help="Deprecated: ignored; evaluation now runs four modes only")
    ap.add_argument("--actor-advantage-design",choices=("full_horizon","rollout_block"),default="full_horizon")
    ap.add_argument("--actor-advantage-steps",type=int,default=1,
        help="1 keeps TD0; 20 collects fixed-Actor blocks and updates with mean n-step advantages")
    ap.add_argument("--actor-diagnostics",action="store_true",
        help="Save additional Full adapt Actor diagnostic plot; no extra modes or post-adaptation runs")
    ap.add_argument("--adapt-start",choices=("initial_heads","fresh","trained"),
                    default="initial_heads",
                    help="Default: retain learned hidden layers, reset Q/R heads and P to fixed initial values")
    ap.add_argument("--save-data",action="store_true",help="Also save CSV, settings and adapted checkpoints; default is images only")
    ap.add_argument("--sensor-noise",action="store_true",
        help="Enable illustrative iid Gaussian noise on all six measured states")
    ap.add_argument("--sensor-std",type=float,nargs=6,default=None,
        metavar=("VX","VY","PSI_RAD","R_RADPS","X","Y"),
        help="Six sensor standard deviations in physical state units; default: 0.005 0.005 0.00872665 0.01 0.005 0.005")
    ap.add_argument("--sensor-seed",type=int,default=10000,
        help="Independent sensor RNG stream: base + evaluation seed, shared by all modes")
    args=ap.parse_args(argv)
    if args.sensor_std is not None and not args.sensor_noise:
        ap.error("--sensor-std requires --sensor-noise")
    if args.sensor_seed<0:
        ap.error("--sensor-seed must be nonnegative")
    noise_std=EXAMPLE_STD if args.sensor_std is None else tuple(args.sensor_std)
    if (len(noise_std)!=6 or not np.isfinite(noise_std).all() or
        np.any(np.asarray(noise_std)<0)):
        ap.error("--sensor-std must be six finite nonnegative values")
    if args.adapt_start=="trained":
        ap.error("Custom evaluation bounds require --adapt-start initial_heads or fresh")
    if args.actor_advantage_steps<1:ap.error("--actor-advantage-steps must be positive")
    if args.max_steps<1 or len(set(args.seeds))!=len(args.seeds):ap.error("Invalid steps/seeds")
    if args.output.exists() and any(args.output.iterdir()):ap.error("Output must be empty")
    # Validate checkpoint before staging. Saved bounds, not editable config.
    _,_,config,_,_=load_checkpoint(args.checkpoint,args.seeds[0])
    config["actor_advantage_steps"]=args.actor_advantage_steps;config["actor_advantage_design"]=args.actor_advantage_design
    args.output.mkdir(parents=True,exist_ok=True)
    staging=tempfile.TemporaryDirectory(prefix="nmpc-evaluation-")
    details,staged=prepare_cad_evaluation_reference(args.reference,Path(staging.name),b.TAIL_LENGTH_M,
        min_speed_mps=float(b.cfg["vx_min"]),
        max_speed_mps=min(float(b.TEST_SPEED_MPS),float(b.cfg["vx_max"])),
        wheelbase_m=float(b.WHEELBASE_M),
        steering_limit_rad=min(abs(float(b.cfg["u_min"][1])),abs(float(b.cfg["u_max"][1]))))
    b.REF_INFO=details;b.TAIL_FILE=Path(staged["tail_file"]);b.CAD_S_END=float(details["cad_s_end"])
    manifest={"checkpoint":str(args.checkpoint.resolve()),
        "checkpoint_sha256":sha256_file(args.checkpoint),
        "reference_sha256":staged["reference_sha256"],
        "same_as_train":staged["reference_sha256"]==TRAIN_REFERENCE_SHA256,
        "fixed_definition":"initial_Q_R_and_separate_fixed_Qf",
        "modes":list(MODES),
        "actor_diagnostics_definition":"extra Full adapt diagnostic plot only",
        "evaluation_role":"online_four_modes_only",
        "sensor_measurements":{
            "enabled":bool(args.sensor_noise),"std_physical_units":list(noise_std),
            "order":["vx_mps","vy_mps","psi_rad","r_radps","X_m","Y_m"],
            "seed_by_evaluation_seed":"sensor_seed + evaluation_seed",
            "base_seed":args.sensor_seed,
            "same_sensor_noise_by_control_step_across_modes":True,
            "plant_dynamics_use_true_state":True,
            "rl_observation_and_td_use_measured_state":True,
            "reported_cte_and_reward_use_true_plant":True,
            "type":"independent_white_gaussian_experimental_not_sensor_calibration"},
        "cte_plot_definition":"absolute CTE over all logged steps including initial transient",
        "settings":vars(args)|{"checkpoint":str(args.checkpoint),"reference":str(args.reference),"output":str(args.output)},
        "checkpoint_config":config,
        "interpretation":"Adaptation learns on this route; do not use it to tune held-out test settings."}
    if args.save_data:
        (args.output/"evaluation_config.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    records=[]
    for seed in args.seeds:
        traces={}
        for mode in MODES:
            actor,critic,config,low,high=load_checkpoint(args.checkpoint,seed)
            config["actor_advantage_steps"]=args.actor_advantage_steps;config["actor_advantage_design"]=args.actor_advantage_design
            if mode=="fixed":
                actor,critic=initial_fixed_policy(config,low,high)
            elif mode=="frozen":
                actor,critic=initialize_adaptation(actor,critic,config,low,high,"initial_heads")
            elif mode in ADAPTING_MODES+("frozen_sampled",):
                actor,critic=initialize_adaptation(actor,critic,config,low,high,args.adapt_start)
            terminal_init=config["fixed_Qf"] if mode=="fixed" else config["initial_P"]
            env=TrainEnvironment(train_length_m=b.CAD_S_END,
                q_initial=np.array(config["initial_Q"]),r_initial=np.array(config["initial_R"]),
                p_initial=np.array(terminal_init),
                task_qe=np.array(config["Qe_fixed"]),task_re=np.array(config["Re_fixed"]),
                max_steps=args.max_steps,ey=args.ey,heading_deg=args.heading_deg,
                sensor=(GaussianStateSensor(noise_std,args.sensor_seed+seed)
                        if args.sensor_noise else None))
            summary,log=run_mode(env,actor,critic,config,mode)
            summary["seed"]=seed
            summary["sensor_noise_enabled"]=bool(args.sensor_noise)
            summary["sensor_noise_seed"]=(args.sensor_seed+seed if args.sensor_noise else None)
            records.append(summary);traces[mode]=log
            if args.save_data:
                folder=args.output/f"seed_{seed}"/mode;folder.mkdir(parents=True)
                log.to_csv(folder/"trajectory.csv",index=False)
                if mode in ADAPTING_MODES:
                    write_checkpoint(folder/"checkpoint_after.npz",actor,critic,
                        {**config,"adaptation_mode":mode,"adaptation_seed":seed,
                         "adaptation_start":args.adapt_start,
                         "adaptation_reference_sha256":staged["reference_sha256"]})
            print(json.dumps(summary),flush=True)
            if args.save_data:pd.DataFrame(records).to_csv(args.output/"comparison.csv",index=False)
        plot_comparison(traces,args.output/f"seed_{seed}")
        if args.actor_diagnostics:
            plot_actor_diagnostics(traces["full_adapt"],args.output/f"seed_{seed}")
    frame=pd.DataFrame(records)
    columns=["total_reward","mean_cte_all_m","max_cte_all_m","mean_cte_post_m",
             "solver_success_pct","completion_pct","loop_p95_ms","reached_end","failure_terminal"]
    if args.save_data:
        frame.groupby("mode")[columns].agg(["mean","std"]).to_csv(args.output/"summary_by_mode.csv")
    staging.cleanup()


if __name__=="__main__":main()
