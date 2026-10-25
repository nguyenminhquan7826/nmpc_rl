"""Continue joint learning on a rollout; this is not frozen evaluation.

Uses stochastic actions: the existing Gaussian score-function update requires
sampled latents. The direct-P Critic has six learned parameters, not a NN.
"""
from __future__ import annotations

import argparse
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd

import baseline as b
from actor_critic_updates import learn_one_transition
from nstep_actor import block_advantages, update_actor_block
from experiment_d_penalties import compute_training_reward
from evaluate_frozen_actor_critic import load_frozen_agents, metrics_from_rows
from train_actor_critic import write_checkpoint


def adaptive_rollout(env, actor, critic, config, *, update_actor=True,
                     update_critic=True, sample_actions=True):
    """Update Critic each transition; Actor uses TD0 or fixed-policy rollout blocks."""
    horizon=config.get("actor_advantage_steps",1)
    if isinstance(horizon,bool) or not isinstance(horizon,(int,np.integer)) or horizon<1:
        raise ValueError("actor_advantage_steps must be a positive integer")
    delayed=bool(update_actor and horizon>1)
    obs=env.reset()
    rows=[];pending=[];p_snapshot=None;actor_updates=0;actor_samples=0

    for _ in range(env.max_steps):
        loop_start=time.perf_counter()
        if update_actor and not sample_actions:
            raise ValueError("Actor score-function learning requires sampled actions")
        proposal=actor.propose(obs.features,deterministic=not sample_actions)
        # Same pre-update observation/policy; deterministic outputs are not E[Q/R].
        q_det,r_det=actor.physical_diagonals(proposal.mu)
        p_before=critic.p.copy()
        if delayed and not pending:p_snapshot=p_before.copy()
        nxt,reward,terminated,truncated,diag=env.step(
            proposal.q_diagonal,proposal.r_diagonal,critic,obs)
        breakdown=compute_training_reward(
            reward,diag,
            solver_failure_penalty=config.get("train_solver_failure_penalty",0.),
            failure_terminal_penalty=config.get("train_failure_terminal_penalty",0.))
        update=learn_one_transition(
            actor,critic,obs.features,proposal.latent,obs.error,nxt.error,
            breakdown.training_reward,gamma=config["gamma"],
            actor_lr=config["actor_lr"],critic_lr=config["critic_lr"],
            terminated=terminated,
            actor_output_scales=config.get("actor_output_lr_factors"),
            critic_component_scales=config.get("critic_p_lr_factors"),
            critic_update_mode=config.get("critic_update_mode","semi_gradient"),
            actor_mu_reg_coef=config.get("actor_mu_reg_coef",0.),
            actor_mu_reg_threshold=config.get("actor_mu_reg_threshold",2.),
            update_actor=update_actor and not delayed,update_critic=update_critic)
        row={**diag,"controller":"online_adaptation",
             "t_s":len(rows)*b.Ts,"X_m":float(obs.physical[4]),
             "Y_m":float(obs.physical[5]),
             "X_ref_m":float(obs.reference["xref"][0][4]),
             "Y_ref_m":float(obs.reference["xref"][0][5]),
             "s_progress_m":float(obs.reference["projection"]["s"]),
             "cross_track_m":float(obs.reference["projection"]["distance"]),
             "e_psi":float(obs.error[2]),"e_vx":float(obs.error[0]),
             "reward":reward,"training_reward":breakdown.training_reward,
             "terminated":terminated,"truncated":truncated,
             "td":update.td,"actor_param_step_norm":update.actor_param_step_norm,
             "critic_param_step_norm":update.critic_param_step_norm}
        for prefix,values in (("q",proposal.q_diagonal),("r",proposal.r_diagonal),
                              ("p",p_before),("p_after",critic.p),
                              ("sigma",proposal.sigma),("mu",proposal.mu),
                              ("latent",proposal.latent),("q_det",q_det),("r_det",r_det),
                              ("obs",obs.features)):
            row.update({f"{prefix}_{i}":float(v) for i,v in enumerate(values)})
        row.update(actor_advantage=(update.td if update_actor and not delayed else np.nan),
                   actor_target=(update.td+update.value_current if update_actor and not delayed else np.nan),
                   actor_advantage_horizon=(1 if update_actor and not delayed else 0),
                   actor_batch_size=(1 if update_actor and not delayed else 0),
                   actor_update_applied=bool(update_actor and not delayed))
        rows.append(row)
        if delayed:
            pending.append((obs.features.copy(),proposal.latent.copy(),obs.error.copy(),
                            breakdown.training_reward,row))
            if len(pending)>=horizon or terminated or truncated or len(rows)>=env.max_steps:
                advantages,targets=block_advantages(
                    [v[3] for v in pending],[v[2] for v in pending],nxt.error,
                    p_snapshot,config["gamma"],terminated)
                loss,step_norm=update_actor_block(actor,[v[0] for v in pending],
                    [v[1] for v in pending],advantages,config)
                for i,entry in enumerate(pending):
                    entry[4].update(actor_advantage=float(advantages[i]),actor_target=float(targets[i]),
                                    actor_advantage_horizon=len(pending)-i)
                row.update(actor_param_step_norm=step_norm,actor_batch_size=len(pending),
                           actor_update_applied=True,actor_batch_loss=loss)
                actor_updates+=1;actor_samples+=len(pending);pending.clear()
        elif update_actor:
            actor_updates+=1;actor_samples+=1
        row.update({f"critic_gradient_{i}":float(v)
                    for i,v in enumerate(update.critic_gradient)})
        row["loop_ms"]=1000*(time.perf_counter()-loop_start)
        obs=nxt
        if terminated or truncated:
            break
    log=pd.DataFrame(rows)
    summary=metrics_from_rows(log,config["gamma"],b.CAD_S_END)
    summary.update(mode="ONLINE_ADAPTATION",parameter_updates=len(log) if update_critic else actor_updates,
                   actor_updates=actor_updates,actor_training_samples=actor_samples,
                   actor_advantage_steps=horizon,
                   actor_update_rule="rollout_block_mean_gradient" if horizon>1 else "TD0",
                   critic_updates=len(log) if update_critic else 0,
                   critic_update_mode=config.get("critic_update_mode","semi_gradient"),
                   action_mode="sampled_gaussian" if sample_actions else "deterministic_mean",
                   reached_end=bool(log.reached_end.iloc[-1]),
                   failure_terminal=bool(log.failure_terminal.iloc[-1]),
                   terminated=bool(log.terminated.iloc[-1]),
                   truncated=bool(log.truncated.iloc[-1]))
    return summary,log


def plot_online(log, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(output)
    fig,axes=plt.subplots(3,1,figsize=(12,12),sharex=True)
    for ax,prefix,labels in (
        (axes[0],"q",("vx","vy","psi","r","X","Y")),
        (axes[1],"r",("acceleration","steering")),
        (axes[2],"p",("vx","vy","psi","r","X","Y"))):
        for i,label in enumerate(labels):
            ax.plot(log.s_progress_m,log[f"{prefix}_{i}"],label=label)
        ax.set_ylabel({"q":"Q diagonal","r":"R diagonal","p":"Qf applied"}[prefix])
        ax.legend(ncol=3);ax.grid(alpha=.25)
    axes[-1].set_xlabel("Path progress (m)")
    fig.suptitle("Online adaptation: applied Actor weights and Critic terminal cost")
    fig.tight_layout();fig.savefig(root/"online_weights.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,7))
    ax.plot(log.X_ref_m,log.Y_ref_m,"k--",label="Reference")
    ax.plot(np.r_[log.X_m,log.X_next_m.iloc[-1]],
            np.r_[log.Y_m,log.Y_next_m.iloc[-1]],label="Online adaptation")
    ax.set(xlabel="X (m)",ylabel="Y (m)");ax.axis("equal");ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(root/"online_trajectory.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for i,label in enumerate(("vx","vy","psi","r","X","Y")):
        ax.plot(log.s_progress_m,log[f"p_after_{i}"]-log[f"p_{i}"].iloc[0],label=label)
    ax.set(xlabel="Path progress (m)",ylabel="P after update minus initial P",
           title="Critic changes (a flat curve can mean projection at a bound)")
    ax.legend(ncol=3);ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(root/"online_qf_change.png",dpi=180);plt.close(fig)


def run_from_checkpoint(env, checkpoint, output, seed=0):
    """Load a separate copy; never overwrite frozen/training checkpoints."""
    root=Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Online output is not empty: {root}")
    with np.load(checkpoint,allow_pickle=False) as saved:
        config=json.loads(str(saved["metadata"].item()))
    low=tuple(np.asarray(config[f"{k}_lower"],float) for k in ("Q","R","P"))
    high=tuple(np.asarray(config[f"{k}_upper"],float) for k in ("Q","R","P"))
    _,(actor,critic),_=load_frozen_agents(checkpoint,low,high,expected_gamma=config["gamma"])
    # A new, explicitly recorded sampling stream; not exact training resume.
    actor.rng=np.random.default_rng(seed)
    root.mkdir(parents=True,exist_ok=True)
    summary,log=adaptive_rollout(env,actor,critic,config)
    log.to_csv(root/"online_episode.csv",index=False)
    summary.update(source_checkpoint=str(Path(checkpoint).resolve()),seed=seed,
                   reference_role="TRAIN_ONLY",not_frozen_evaluation=True)
    (root/"online_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    write_checkpoint(root/"checkpoint_after_online.npz",actor,critic,
                     {**config,"adaptation_mode":"ONLINE_ADAPTATION",
                      "adaptation_seed":seed,"adaptation_steps":len(log),
                      "source_checkpoint":str(Path(checkpoint).resolve())})
    plot_online(log,root)
    return summary


def main(argv=None):
    from cad_eval_reference import prepare_cad_evaluation_reference
    from screen_training_weights import verify_train_reference
    from train_rl_environment import TrainEnvironment
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--reference",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--max-steps",type=int,default=1150)
    ap.add_argument("--seed",type=int,default=0)
    args=ap.parse_args(argv)
    if args.max_steps<1:ap.error("--max-steps must be positive")
    if args.output.exists() and any(args.output.iterdir()):ap.error("Output must be empty")
    ref,digest=verify_train_reference(args.reference)
    with np.load(args.checkpoint,allow_pickle=False) as saved:
        config=json.loads(str(saved["metadata"].item()))
    # Stage separately from output (which run_from_checkpoint protects).
    import tempfile
    with tempfile.TemporaryDirectory(prefix="nmpc-online-") as tmp:
        details,staged=prepare_cad_evaluation_reference(
            ref,Path(tmp),b.TAIL_LENGTH_M,
            min_speed_mps=float(b.cfg["vx_min"]),
            max_speed_mps=min(float(b.TEST_SPEED_MPS),float(b.cfg["vx_max"])),
            wheelbase_m=float(b.WHEELBASE_M),
            steering_limit_rad=min(abs(float(b.cfg["u_min"][1])),abs(float(b.cfg["u_max"][1]))))
        if staged["reference_sha256"]!=digest:raise RuntimeError("Reference changed")
        b.REF_INFO=details;b.TAIL_FILE=Path(staged["tail_file"]);b.CAD_S_END=float(details["cad_s_end"])
        env=TrainEnvironment(train_length_m=b.CAD_S_END,
            q_initial=np.array(config["initial_Q"]),r_initial=np.array(config["initial_R"]),
            p_initial=np.array(config["initial_P"]),task_qe=np.array(config["Qe_fixed"]),
            task_re=np.array(config["Re_fixed"]),max_steps=args.max_steps,
            ey=config.get("ey_m",.05),heading_deg=config.get("heading_deg",0.))
        print(json.dumps(run_from_checkpoint(env,args.checkpoint,args.output,args.seed)),flush=True)


if __name__=="__main__":
    main()
