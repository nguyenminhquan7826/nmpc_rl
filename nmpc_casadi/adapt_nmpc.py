"""Paired online adaptation for rl_nmpc v7/v8; preserves user's controller configuration.
Run: python adapt_nmpc.py --checkpoint TRAIN/checkpoint.npz --output adapt_results
Online Actor uses sampled actions: deterministic actions cannot provide its score gradient.
"""
import argparse
import inspect
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import rl_nmpc as rl


def fresh_agent(args, meta):
    kw=dict(seed=args.seed, sigma=meta.get('sigma',rl.SIGMA),
            actor_lr=args.actor_lr, critic_lr=args.critic_lr,
            critic_nsteps=meta.get('critic_nsteps',16))
    if 'train_component' in inspect.signature(rl.Agent).parameters:
        kw.update(train_component=meta.get('train_component','both'),
                  initial_seed=meta.get('initial_seed'))
    agent=rl.Agent(**kw)
    agent.load(args.checkpoint)
    # Same checkpoint optimizer and policy RNG for every scenario and paired run.
    return agent


def run(env, agent, args, scenario, variant):
    name,ey,heading,plant=scenario
    online=variant=='online'
    explore=variant!='frozen'
    original=agent.learn
    timing=[]
    def timed_learn(rows):
        start=time.perf_counter()
        result=original(rows)
        elapsed=(time.perf_counter()-start)*1000
        timing.append(elapsed)
        return {**result,'learn_ms':elapsed}
    agent.learn=timed_learn
    label='critic_only' if getattr(agent,'train_component','both')=='critic' else 'actor_critic'
    start=time.perf_counter()
    _,m=rl.rollout(env,agent,explore,ey,heading,plant=plant,train=online,
                  rollout_steps=args.rollout_steps,controller=label)
    wall=(time.perf_counter()-start)*1000
    logs=pd.DataFrame(env.logs)
    # Updates happen after the action at the logged step. Remaining steps use new parameters.
    logs['learn_ms']=0.0
    logs['updates_before_action']=0
    updates=getattr(env,'learning_updates',[]) if online else []
    for update in updates:
        k=int(update['step'])
        logs.loc[k-1,'learn_ms']=update['learn_ms']
        logs.loc[k:,'updates_before_action']+=1
    logs['solve_plus_learn_ms']=logs.solve_ms+logs.learn_ms
    logs.to_csv(args.output/f'{name}_{variant}.csv',index=False)
    if online:
        pd.DataFrame(updates).to_csv(args.output/f'{name}_updates.csv',index=False)
        agent.save(args.output/f'{name}_adapted_checkpoint.npz')
    m.update(scenario=name,controller=variant,updates=len(updates),
             learning_total_ms=float(sum(timing)),
             learning_p95_ms=float(np.quantile(timing,.95)) if timing else 0.,
             solve_plus_learn_p95_ms=float(logs.solve_plus_learn_ms.quantile(.95)),
             solve_plus_learn_max_ms=float(logs.solve_plus_learn_ms.max()),
             solve_plus_learn_over_Ts_pct=float(100*(logs.solve_plus_learn_ms>rl.b.Ts*1000).mean()),
             rollout_wall_ms=wall)
    return m


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--rollout-steps',type=int,default=100)
    ap.add_argument('--actor-lr',type=float,default=.001)
    ap.add_argument('--critic-lr',type=float,default=.003)
    ap.add_argument('--seconds',type=float,default=120)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--nominal-only',action='store_true')
    args=ap.parse_args()
    if args.rollout_steps<1 or args.seconds<=0 or not np.isfinite(args.seconds):
        ap.error('rollout-steps and seconds must be positive')
    for lr in [args.actor_lr,args.critic_lr]:
        if not np.isfinite(lr) or lr<=0:ap.error('learning rates must be positive and finite')
    with np.load(args.checkpoint,allow_pickle=False) as saved:
        meta=json.loads(str(saved['metadata'].item()))
    fresh_agent(args,meta)  # Validate strict configuration before creating outputs.
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()):ap.error('Use an empty output folder to preserve prior results')
    (args.output/'config.json').write_text(json.dumps(dict(mode='adapt',checkpoint_metadata=meta,
        checkpoint_source=str(args.checkpoint),rollout_steps=args.rollout_steps,
        actor_lr=args.actor_lr,critic_lr=args.critic_lr,seconds=args.seconds,
        variants=['frozen','frozen_explore','online'],reset_checkpoint_each_run=True,
        timing_note='solve_plus_learn excludes observation, NN inference and file IO; wall time includes rollout'),indent=2))
    scenarios=[('nominal',.05,0,rl.b.p_nominal.copy())]
    if not args.nominal_only:
        for ey,deg in [(-.05,5),(.05,-5),(.1,-10)]:
            scenarios.append((f'ey{ey}_psi{deg}',ey,np.deg2rad(deg),rl.b.p_nominal.copy()))
        for index,name in [(0,'mass'),(1,'Iz')]:
            for factor in [.9,1.1]:
                plant=rl.b.p_nominal.copy();plant[index]*=factor
                scenarios.append((f'{name}_{factor}',.05,0,plant))
    env=rl.Environment(args.seconds)
    results=[]
    for scenario in scenarios:
        for variant in ['frozen','frozen_explore','online']:
            m=run(env,fresh_agent(args,meta),args,scenario,variant)
            results.append(m)
            pd.DataFrame(results).to_csv(args.output/'comparison.csv',index=False)
            print(json.dumps(m),flush=True)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for variant in ['frozen','frozen_explore','online']:
        log=pd.read_csv(args.output/f'nominal_{variant}.csv')
        axes[0].plot(log.t_s,log.cte_m*100,label=variant)
        axes[1].plot(log.t_s,log.progress_m/rl.b.CAD_S_END*100,label=variant)
    axes[0].set_ylabel('CTE [cm]');axes[1].set_ylabel('Completion [%]');axes[1].set_xlabel('Time [s]')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(args.output/'online_comparison.png',dpi=160);plt.close(fig)
    log=pd.read_csv(args.output/'nominal_online.csv')
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for ax,prefix in zip(axes,['q','r','qf']):
        for col in log:
            if col.startswith(prefix+'_'):ax.plot(log.t_s,log[col],label=col)
        ax.set_ylabel(prefix);ax.legend(ncol=3);ax.grid(alpha=.2)
    axes[-1].set_xlabel('Time [s]');fig.tight_layout();fig.savefig(args.output/'online_weights.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
