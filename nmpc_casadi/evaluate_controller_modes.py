"""Compare five controller modes from independent copies of one checkpoint.

Adaptation modes learn on the supplied route and are not frozen test scores.
The optional frozen_sampled control isolates sampling from parameter updates.
"""
import argparse
import json
import os
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
from training_initialization import GOOD_NMPC, TRAIN_REFERENCE_SHA256
from run_experiment import FixedWeightActor

MODES=("fixed","frozen","actor_adapt","critic_adapt","full_adapt")


def load_checkpoint(path, seed):
    with np.load(path,allow_pickle=False) as z:
        config=json.loads(str(z["metadata"].item()))
    low=tuple(np.array(config[f"{k}_lower"],float) for k in ("Q","R","P"))
    high=tuple(np.array(config[f"{k}_upper"],float) for k in ("Q","R","P"))
    _,(actor,critic),_=load_frozen_agents(path,low,high,expected_gamma=config["gamma"])
    actor.rng=np.random.default_rng(seed)
    return actor,critic,config,low,high


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
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    first=next(iter(traces.values()))
    axes[0].plot(first.X_ref_m,first.Y_ref_m,"k--",label="Reference (logged)")
    for mode,log in traces.items():
        x=np.r_[log.X_m,log.X_next_m.iloc[-1]]
        y=np.r_[log.Y_m,log.Y_next_m.iloc[-1]]
        axes[0].plot(x,y,label=mode,alpha=.85)
        axes[1].plot(log.s_progress_m,100*log.cross_track_m,label=mode,alpha=.85)
    axes[0].set(xlabel="X (m)",ylabel="Y (m)");axes[0].axis("equal")
    axes[1].set(xlabel="Path progress (m)",ylabel="CTE (cm)")
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(Path(output)/"mode_comparison.png",dpi=180);plt.close(fig)


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
    args=ap.parse_args(argv)
    if args.max_steps<1 or len(set(args.seeds))!=len(args.seeds):ap.error("Invalid steps/seeds")
    if args.output.exists() and any(args.output.iterdir()):ap.error("Output must be empty")
    # Validate checkpoint before staging. Saved bounds, not editable config.
    _,_,config,_,_=load_checkpoint(args.checkpoint,args.seeds[0])
    args.output.mkdir(parents=True,exist_ok=True)
    details,staged=prepare_cad_evaluation_reference(args.reference,args.output/"reference",b.TAIL_LENGTH_M,
        min_speed_mps=float(b.cfg["vx_min"]),
        max_speed_mps=min(float(b.TEST_SPEED_MPS),float(b.cfg["vx_max"])),
        wheelbase_m=float(b.WHEELBASE_M),
        steering_limit_rad=min(abs(float(b.cfg["u_min"][1])),abs(float(b.cfg["u_max"][1]))))
    b.REF_INFO=details;b.TAIL_FILE=Path(staged["tail_file"]);b.CAD_S_END=float(details["cad_s_end"])
    manifest={"checkpoint":str(args.checkpoint.resolve()),
        "checkpoint_sha256":sha256_file(args.checkpoint),
        "reference_sha256":staged["reference_sha256"],
        "same_as_train":staged["reference_sha256"]==TRAIN_REFERENCE_SHA256,
        "settings":vars(args)|{"checkpoint":str(args.checkpoint),"reference":str(args.reference),"output":str(args.output)},
        "checkpoint_config":config,
        "interpretation":"Adaptation learns on this route; do not use it to tune held-out test settings."}
    (args.output/"evaluation_config.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    records=[]
    for seed in args.seeds:
        traces={}
        for mode in MODES+(("frozen_sampled",) if args.sampling_control else ()):
            actor,critic,config,low,high=load_checkpoint(args.checkpoint,seed)
            if mode=="fixed":
                q,r,p=GOOD_NMPC.arrays()
                actor=FixedWeightActor(q,r)
                critic=SixParameterTDCritic(p,low[2],high[2])
            env=TrainEnvironment(train_length_m=b.CAD_S_END,
                q_initial=np.array(config["initial_Q"]),r_initial=np.array(config["initial_R"]),
                p_initial=np.array(config["initial_P"]),
                task_qe=np.array(config["Qe_fixed"]),task_re=np.array(config["Re_fixed"]),
                max_steps=args.max_steps,ey=args.ey,heading_deg=args.heading_deg)
            summary,log=run_mode(env,actor,critic,config,mode)
            summary["seed"]=seed;records.append(summary);traces[mode]=log
            folder=args.output/f"seed_{seed}"/mode;folder.mkdir(parents=True)
            log.to_csv(folder/"trajectory.csv",index=False)
            if mode in ("actor_adapt","critic_adapt","full_adapt"):
                write_checkpoint(folder/"checkpoint_after.npz",actor,critic,
                    {**config,"adaptation_mode":mode,"adaptation_seed":seed,
                     "adaptation_reference_sha256":staged["reference_sha256"]})
            print(json.dumps(summary),flush=True)
            pd.DataFrame(records).to_csv(args.output/"comparison.csv",index=False)
        plot_comparison(traces,args.output/f"seed_{seed}")
    frame=pd.DataFrame(records)
    columns=["total_reward","mean_cte_all_m","max_cte_all_m","mean_cte_post_m",
             "solver_success_pct","completion_pct","loop_p95_ms","reached_end","failure_terminal"]
    frame.groupby("mode")[columns].agg(["mean","std"]).to_csv(args.output/"summary_by_mode.csv")


if __name__=="__main__":main()
