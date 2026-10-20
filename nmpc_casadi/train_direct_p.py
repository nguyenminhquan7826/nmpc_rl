"""Train/evaluate Direct-P Critic without modifying legacy RL source.

Usage (from nmpc_casadi):
  python train_direct_p.py --mode smoke --seconds 3 --rollout-steps 1 --output results_direct_p/smoke
  python train_direct_p.py --mode train --episodes 100 --seconds 120 --rollout-steps 1 --output results_direct_p/train_seed0
  python train_direct_p.py --mode evaluate --checkpoint results_direct_p/train_seed0/checkpoint.npz --output results_direct_p/eval_seed0

train_component=critic preserves Actor parameters while still sampling the
frozen Actor; both updates Actor and Critic. Frozen uses the same initial P.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import rl_nmpc as rl
from direct_p_critic import DirectPAgent

CONTROLLERS = ('fixed_nmpc', 'direct_p_frozen', 'direct_p_actor_only',
               'direct_p_critic_only', 'direct_p_both')


def make_agent(args):
    agent = DirectPAgent(seed=args.seed, actor_lr=args.actor_lr,
                         critic_lr=args.critic_lr, sigma=args.sigma,
                         p_scale=args.p_scale, p_min_multiplier=args.p_min_multiplier,
                         p_max_multiplier=args.p_max_multiplier)
    if args.checkpoint:
        agent.load(args.checkpoint, sigma_override=args.sigma)
    return agent


def run(env, agent, args, controller, train=False):
    if controller not in CONTROLLERS:
        raise ValueError(controller)
    # Use same starting geometry/plant within comparison, independently reset RNG.
    obs, error = env.reset(args.ey, np.deg2rad(args.heading_deg))
    saved_rng = json.dumps(agent.rng.bit_generator.state)
    batch = []
    metrics = []
    count = 0
    while True:
        weights, latent = agent.propose(obs, explore=controller!='fixed_nmpc')
        q,r,p = weights
        if controller=='fixed_nmpc':
            weights = tuple(v.copy() for v in env.initial_weights)
        elif controller=='direct_p_actor_only':
            weights = (q,r,agent.critic.physical_diagonal().copy())
        elif controller=='direct_p_critic_only':
            weights = (env.initial_weights[0].copy(),
                       env.initial_weights[1].copy(),p)
        next_obs, next_error, reward, done, terminated = env.step(weights)
        if controller!='fixed_nmpc':
            env.logs[-1]['value'] = float(agent.value(obs,error))
            env.logs[-1]['next_value'] = float(agent.value(next_obs,next_error))
            env.logs[-1]['value_b'] = 0.
            e = np.asarray(error)
            energy=float(np.sum(e*e*rl.F_BASE))
            env.logs[-1].update(
                quadratic_min=-args.p_max_multiplier*energy,
                quadratic_max=-args.p_min_multiplier*energy,
                full_value_upper=-args.p_min_multiplier*energy,
                curvature=float(obs[9]*rl.SCALE[9]),
                truncated=int(done and not terminated))
        batch.append((obs.copy(),error.copy(),np.asarray(latent).copy(),
                      reward,next_obs.copy(),next_error.copy(),float(terminated)))
        obs,error=next_obs,next_error
        if train and (len(batch)>=args.rollout_steps or done):
            from ac_ablation import learn_selected_component
            mode = {'direct_p_actor_only':'actor',
                    'direct_p_critic_only':'critic',
                    'direct_p_both':'both'}[controller]
            result=learn_selected_component(agent,batch,mode)
            metrics.append({'step':env.k,'batch_steps':len(batch),**result})
            batch=[]
        if done: break
    summary=rl.metrics(env.logs)
    summary.update(controller=controller,updates=len(metrics),
                   final_p=agent.critic.physical_diagonal().tolist())
    return pd.DataFrame(env.logs), pd.DataFrame(metrics), summary


def save_run(output, label, logs, updates, summary):
    logs.to_csv(output/f'{label}.csv',index=False)
    if len(updates): updates.to_csv(output/f'{label}_updates.csv',index=False)
    return {k:v for k,v in summary.items() if k!='final_p'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['smoke','train','evaluate'],default='smoke')
    ap.add_argument('--train-component',choices=['critic','both','actor'],default='critic')
    ap.add_argument('--episodes',type=int,default=100)
    ap.add_argument('--seconds',type=float,default=120)
    ap.add_argument('--rollout-steps',type=int,default=1)
    ap.add_argument('--eval-every',type=int,default=10)
    ap.add_argument('--actor-lr',type=float,default=1e-3)
    ap.add_argument('--critic-lr',type=float,default=1e-3)
    ap.add_argument('--sigma',type=float,default=.05)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--ey',type=float,default=.05)
    ap.add_argument('--heading-deg',type=float,default=0.)
    ap.add_argument('--p-scale',choices=['physical','normalized'],default='normalized')
    ap.add_argument('--p-min-multiplier',type=float,default=.5)
    ap.add_argument('--p-max-multiplier',type=float,default=50.)
    ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.mode=='evaluate' and args.checkpoint is None:ap.error('Evaluate requires --checkpoint')
    if (args.seconds<=0 or args.episodes<1 or args.rollout_steps<1
            or args.eval_every<1):ap.error('Invalid training horizon')
    if args.output.exists() and any(args.output.iterdir()):
        ap.error('Output folder must be empty; old results are protected')
    args.output.mkdir(parents=True,exist_ok=True)
    env=rl.Environment(args.seconds)
    config={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}
    config.update(critic='direct_diagonal_quadratic',td='one_step_semi_gradient',
                  stage_reward='unchanged_R3A_fixed_eval',Ts=rl.b.Ts,N=rl.b.N)
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    agent=make_agent(args)
    if args.mode=='evaluate':
        controllers=CONTROLLERS[:2]
        outcomes=[]
        for controller in controllers:
            a=make_agent(args)
            logs,updates,summary=run(env,a,args,controller,train=False)
            outcomes.append(save_run(args.output,controller,logs,updates,summary))
        pd.DataFrame(outcomes).to_csv(args.output/'comparison.csv',index=False)
        print(pd.DataFrame(outcomes).to_string(index=False))
        return

    train_controller={'critic':'direct_p_critic_only','actor':'direct_p_actor_only',
                      'both':'direct_p_both'}[args.train_component]
    history=[]
    for ep in range(2 if args.mode=='smoke' else args.episodes):
        logs,updates,summary=run(env,agent,args,train_controller,train=True)
        save_run(args.output,f'episode_{ep:04d}',logs,updates,summary)
        agent.save(args.output/'checkpoint.npz')
        history.append({'episode':ep,**{k:v for k,v in summary.items()
                                        if k!='final_p'},
                        **{f'p_final_{i}':p for i,p in enumerate(summary['final_p'])}})
        pd.DataFrame(history).to_csv(args.output/'training_history.csv',index=False)
        print(json.dumps(history[-1]),flush=True)
        if ep==0 or (ep+1)%args.eval_every==0 or ep==args.episodes-1:
            rows=[]
            for controller in ('fixed_nmpc','direct_p_frozen'):
                fresh=make_agent(args)
                # current policy snapshot for deterministic evaluation at this episode
                fresh.actor.p=[x.copy() for x in agent.actor.p]
                fresh.critic.p=[x.copy() for x in agent.critic.p]
                log,upd,m=run(env,fresh,args,controller,train=False)
                rows.append(save_run(args.output,f'validation_ep{ep:04d}_{controller}',log,upd,m))
            pd.DataFrame(rows).to_csv(args.output/f'validation_ep{ep:04d}.csv',index=False)

if __name__=='__main__':
    main()
