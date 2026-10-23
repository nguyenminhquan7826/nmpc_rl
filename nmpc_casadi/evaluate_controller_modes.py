"""Compare five controller modes from independent copies of one checkpoint.

Adaptation modes learn on the supplied route and are not frozen test scores.
The optional frozen_sampled control isolates sampling from parameter updates.
Frozen resets the trained Actor heads and P to initial Q/R/P, then locks them.
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
from training_initialization import TRAIN_REFERENCE_SHA256
from run_experiment import FixedWeightActor

MODES=("fixed","frozen","actor_adapt","critic_adapt","full_adapt")


def initial_fixed_policy(config,low,high):
    q=np.asarray(config["initial_Q"],float)
    r=np.asarray(config["initial_R"],float)
    # Independent terminal configuration: never borrow the adapting P.
    qf=np.asarray(config.get("fixed_Qf",config["initial_P"]),float)
    return FixedWeightActor(q,r),SixParameterTDCritic(qf,low[2],high[2])

def load_checkpoint(path, seed):
    with np.load(path, allow_pickle=False) as z:
        config = json.loads(str(z["metadata"].item()))

    low = tuple(
        np.array(config[f"{k}_lower"], dtype=float)
        for k in ("Q", "R", "P")
    )
    high = tuple(
        np.array(config[f"{k}_upper"], dtype=float)
        for k in ("Q", "R", "P")
    )

    # Kiểm tra checkpoint bằng biên training gốc trước.
    _, (actor, critic), _ = load_frozen_agents(
        path, low, high,
        expected_gamma=config["gamma"],
    )
    # Miền thử nghiệm minh họa — chưa được kiểm chứng điều khiển.
    # Q: vx, vy, psi, r, X, Y
    q_lower = np.array([1, 0.2, 1, 0.5, 10, 10], dtype=float)
    q_upper = np.array([100, 50, 250, 100, 500, 500], dtype=float)

    # R: gia tốc, góc lái
    r_lower = np.array([0.05, 0.5], dtype=float)
    r_upper = np.array([30, 800], dtype=float)

    # P/Qf: vx, vy, psi, r, X, Y
    p_lower = np.array([10, 2, 20, 5, 100, 100], dtype=float)
    p_upper = np.array([1000, 300, 2000, 500, 25000, 25000],
                    dtype=float)

    low = (q_lower, r_lower, p_lower)
    high = (q_upper, r_upper, p_upper)

    # Đồng bộ metadata.
    for key, lower, upper in zip(("Q", "R", "P"), low, high):
        if (
            not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or np.any(lower <= 0)
            or np.any(lower >= upper)
        ):
            raise ValueError(f"Invalid eval bounds: {key}")

        config[f"{key}_lower"] = lower.tolist()
        config[f"{key}_upper"] = upper.tolist()

    # Đồng bộ miền thực tế của model.
    actor.lower = np.r_[q_lower, r_lower].copy()
    actor.upper = np.r_[q_upper, r_upper].copy()
    critic.lower = p_lower.copy()
    critic.upper = p_upper.copy()
    # Bộ khởi tạo chung cho các mode reset.
    config["initial_Q"] = [17, 4, 70, 3.1, 51.0, 51.0]
    config["initial_R"] = [15, 200]

    config["fixed_Qf"] = (
        15.0 * np.asarray(config["initial_Q"], dtype=float)
    ).tolist()
    config["initial_P"] = config["fixed_Qf"].copy()

    # Miền R mới: initial phải nằm nghiêm ngặt trong biên.
    r_upper = np.array([20.0, 300.0])
    high = (high[0], r_upper, high[2])

    # Kiểm tra và lưu biên eval.
    for key, lower, upper in zip(("Q", "R", "P"), low, high):
        initial = np.asarray(config[f"initial_{key}"], dtype=float)

        if (
            initial.shape != lower.shape
            or upper.shape != lower.shape
            or not np.isfinite(initial).all()
            or not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or np.any(lower <= 0)
            or np.any(lower >= upper)
        ):
            raise ValueError(f"Invalid eval weights/bounds: {key}")

        valid = (
            (initial > lower) & (initial < upper)
            if key in ("Q", "R")
            else (initial >= lower) & (initial <= upper)
        )

        if not np.all(valid):
            raise ValueError(
                f"initial_{key} outside eval bounds:\n"
                f"initial={initial}\nlower={lower}\nupper={upper}"
            )

        config[f"{key}_lower"] = lower.tolist()
        config[f"{key}_upper"] = upper.tolist()

    # Đồng bộ cả miền thực tế của Actor/Critic.
    actor.lower = np.r_[low[0], low[1]].copy()
    actor.upper = np.r_[high[0], high[1]].copy()
    critic.lower = low[2].copy()
    critic.upper = high[2].copy()
    actor.rng = np.random.default_rng(seed)

    config["actor_lr"] = 1e-5
    config["critic_lr"] = 20000.0

    config["actor_output_lr_factors"] = [
        1.0, 1.0, 0.5, 0.5, 2.0, 2.0, 0.5, 0.5
    ]
    config["critic_p_lr_factors"] = [
        0.5, 0.5, 0.1, 0.05, 4.0, 4.0
    ]

    return actor, critic, config, low, high


def initialize_adaptation(actor,critic,config,low,high,start):
    """Reset nominal policy outputs/P without changing input checkpoint."""
    if start=="trained":return actor,critic
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
               "full_adapt":(True,True,True),
               "frozen_sampled":(False,False,True)}
        if mode not in flags:raise ValueError("Unknown controller mode")
        ua,uc,sample=flags[mode]
        summary,log=adaptive_rollout(env,actor,critic,config,
            update_actor=ua,update_critic=uc,sample_actions=sample)
    log["controller"]=mode
    summary.update(mode=mode,controller=mode,
        score_role="online_adaptation" if mode.endswith("adapt") else "frozen_evaluation",
        loop_p95_ms=float(np.percentile(log.loop_ms,95)),
        full_loop_deadline_missed_pct=float(100*(log.loop_ms>b.DEADLINE_MS).mean()),
        mean_du_a_sq=float((log.du_a**2).mean()),
        mean_du_delta_sq=float((log.du_delta**2).mean()),
        mean_cte_all_m=float(log.cross_track_m.abs().mean()),
        max_cte_all_m=float(log.cross_track_m.abs().max()))
    return summary,log


def plot_comparison(traces,output):
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
    ax.set(xlabel="X (m)",ylabel="Y (m)",title="Controller trajectories (fixed = initial Q/R/Qf)")
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
        fig.suptitle("Full adapt: applied Q, R and Qf during online learning")
        fig.tight_layout();fig.savefig(output/"full_adapt_weights.png",dpi=180);plt.close(fig)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--reference",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--max-steps",type=int,default=2000)
    ap.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    ap.add_argument("--ey",type=float,default=.05)
    ap.add_argument("--heading-deg",type=float,default=0.)
    ap.add_argument("--sampling-control",action="store_true")
    ap.add_argument("--adapt-start",choices=("initial_heads","fresh","trained"),
                    default="initial_heads",
                    help="Default: retain learned hidden layers, reset Q/R heads and P to fixed initial values")
    ap.add_argument("--save-data",action="store_true",help="Also save CSV, settings and adapted checkpoints; default is images only")
    args=ap.parse_args(argv)
    if args.max_steps<1 or len(set(args.seeds))!=len(args.seeds):ap.error("Invalid steps/seeds")
    if args.output.exists() and any(args.output.iterdir()):ap.error("Output must be empty")
    # Validate checkpoint before staging. Saved bounds, not editable config.
    _,_,config,_,_=load_checkpoint(args.checkpoint,args.seeds[0])
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
        "frozen_definition":"trained_hidden_layers_reset_heads_and_P_to_initial_no_updates",
        "cte_plot_definition":"absolute CTE over all logged steps including initial transient",
        "settings":vars(args)|{"checkpoint":str(args.checkpoint),"reference":str(args.reference),"output":str(args.output)},
        "checkpoint_config":config,
        "interpretation":"Adaptation learns on this route; do not use it to tune held-out test settings."}
    if args.save_data:
        (args.output/"evaluation_config.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    records=[]
    for seed in args.seeds:
        traces={}
        for mode in MODES+(("frozen_sampled",) if args.sampling_control else ()):
            actor,critic,config,low,high=load_checkpoint(args.checkpoint,seed)
            if mode=="fixed":
                actor,critic=initial_fixed_policy(config,low,high)
            elif mode=="frozen":
                actor,critic=initialize_adaptation(actor,critic,config,low,high,"initial_heads")
            elif mode in ("actor_adapt","critic_adapt","full_adapt","frozen_sampled"):
                actor,critic=initialize_adaptation(actor,critic,config,low,high,args.adapt_start)
            terminal_init=config["fixed_Qf"] if mode=="fixed" else config["initial_P"]
            env=TrainEnvironment(train_length_m=b.CAD_S_END,
                q_initial=np.array(config["initial_Q"]),r_initial=np.array(config["initial_R"]),
                p_initial=np.array(terminal_init),
                task_qe=np.array(config["Qe_fixed"]),task_re=np.array(config["Re_fixed"]),
                max_steps=args.max_steps,ey=args.ey,heading_deg=args.heading_deg)
            summary,log=run_mode(env,actor,critic,config,mode)
            summary["seed"]=seed;records.append(summary);traces[mode]=log
            if args.save_data:
                folder=args.output/f"seed_{seed}"/mode;folder.mkdir(parents=True)
                log.to_csv(folder/"trajectory.csv",index=False)
                if mode in ("actor_adapt","critic_adapt","full_adapt"):
                    write_checkpoint(folder/"checkpoint_after.npz",actor,critic,
                        {**config,"adaptation_mode":mode,"adaptation_seed":seed,
                         "adaptation_start":args.adapt_start,
                         "adaptation_reference_sha256":staged["reference_sha256"]})
            print(json.dumps(summary),flush=True)
            if args.save_data:pd.DataFrame(records).to_csv(args.output/"comparison.csv",index=False)
        plot_comparison(traces,args.output/f"seed_{seed}")
    frame=pd.DataFrame(records)
    columns=["total_reward","mean_cte_all_m","max_cte_all_m","mean_cte_post_m",
             "solver_success_pct","completion_pct","loop_p95_ms","reached_end","failure_terminal"]
    if args.save_data:
        frame.groupby("mode")[columns].agg(["mean","std"]).to_csv(args.output/"summary_by_mode.csv")
    staging.cleanup()


if __name__=="__main__":main()
