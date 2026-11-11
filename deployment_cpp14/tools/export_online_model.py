"""Host-only conversion from an NPZ checkpoint to a native C++14 model.

Never imports baseline/evaluation drivers or executes a rollout. Evaluation
settings must be supplied explicitly; no hidden LR/bound overrides.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'nmpc_casadi'))
from gaussian_weight_actor import GaussianWeightActor
from direct_p_critic import SixParameterTDCritic
SHAPES={'W1':(17,64),'b1':(64,),'W2':(64,64),'b2':(64,),'W_mu':(64,8),'b_mu':(8,),'W_sigma':(64,8),'b_sigma':(8,)}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def export(checkpoint,output,start,config_path=None,horizon=None,design=None):
    if output.exists() or Path(str(output)+'.json').exists():raise FileExistsError('Output already exists')
    with np.load(checkpoint,allow_pickle=False) as z:
        source=json.loads(str(z['metadata'].item()));config=source.copy()
        tensors={}
        for k,shape in SHAPES.items():
            a=np.asarray(z['actor_'+k],float)
            if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Invalid tensor '+k)
            tensors[k]=a.copy()
        trained_p=np.asarray(z['critic_physical_P'],float).copy()
    # Validate source ranges even when an evaluation config later replaces them.
    def bounds(c,k,n):
        lo=np.asarray(c[k+'_lower'],float);hi=np.asarray(c[k+'_upper'],float)
        if lo.shape!=(n,) or hi.shape!=(n,) or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo<=0) or np.any(hi<=lo):raise ValueError('Invalid '+k+' bounds')
        return lo,hi
    original={k:bounds(source,k,n) for k,n in [('Q',6),('R',2),('P',6)]}
    SixParameterTDCritic(trained_p,*original['P'])
    if config_path:
        supplied=json.loads(config_path.read_text());config.update(supplied.get('checkpoint_config',supplied))
    limits={k:bounds(config,k,n) for k,n in [('Q',6),('R',2),('P',6)]}
    if start=='trained':
        for k in ('Q','R'):
            if not all(np.array_equal(a,b) for a,b in zip(original[k],limits[k])):raise ValueError('trained cannot remap learned Q/R bounds; use initial_heads')
    q=np.asarray(config['initial_Q'],float);r=np.asarray(config['initial_R'],float)
    qf=np.asarray(config.get('fixed_Qf',config['initial_P']),float)
    p=np.asarray(config['initial_P'],float) if start=='initial_heads' else trained_p
    # Reuse authoritative Python validation and exact inverse-head initialization.
    actor=GaussianWeightActor(np.random.default_rng(0),q,r,*limits['Q'],*limits['R'],sigma_lower=config['sigma_lower'],sigma_initial=config['sigma_initial'],sigma_upper=config['sigma_upper'])
    if start=='initial_heads':
        for k in ('W_mu','b_mu','W_sigma','b_sigma'):tensors[k]=actor.params[k].copy()
    SixParameterTDCritic(p,*limits['P']);SixParameterTDCritic(qf,*limits['P'])
    h=config.get('actor_advantage_steps',1) if horizon is None else horizon
    de=config.get('actor_advantage_design','full_horizon') if design is None else design
    if isinstance(h,bool) or int(h)!=h or not 1<=h<=10000 or de not in ('full_horizon','rollout_block'):raise ValueError('Invalid Actor schedule')
    cm=config.get('critic_update_mode','semi_gradient')
    if cm not in ('semi_gradient','residual_gradient'):raise ValueError('Unknown Critic update mode')
    gamma=float(config['gamma']);lr=float(config['actor_lr']);cr=float(config['critic_lr'])
    factors=np.asarray(config.get('actor_output_lr_factors') or [1]*8,float)
    rates=cr*np.asarray(config.get('critic_p_lr_factors') or [1]*6,float)
    reg=np.array([config.get('actor_mu_reg_coef',0.),config.get('actor_mu_reg_threshold',2.)],float)
    length=float(config['train_length_m'])
    if (not 0<gamma<=1 or not np.isfinite([lr,cr,length]).all() or min(lr,cr,length)<=0 or factors.shape!=(8,) or rates.shape!=(6,) or not np.isfinite(factors).all() or not np.isfinite(rates).all() or np.any(factors<=0) or np.any(rates<=0) or not np.isfinite(reg).all() or reg[0]<0 or reg[1]<=0):raise ValueError('Invalid learning configuration')
    values={**tensors,'lower':np.r_[limits['Q'][0],limits['R'][0]],'upper':np.r_[limits['Q'][1],limits['R'][1]],
            'sigma_bounds':[config['sigma_lower'],config['sigma_upper']],'critic_p':p,'critic_lower':limits['P'][0],'critic_upper':limits['P'][1],
            'fixed_q':q,'fixed_r':r,'fixed_qf':qf,'gamma':[gamma],'actor_lr':[lr],'critic_rates':rates,'actor_factors':factors,'mu_reg':reg,
            'critic_mode':[int(cm=='residual_gradient')],'horizon':[h],'design':[int(de=='rollout_block')],'train_length':[length]}
    lines=['AC_MODEL_V1']
    for k,v in values.items():
        a=np.asarray(v).ravel();lines.append(k+' '+str(a.size)+' '+' '.join(format(float(x),'.17g') for x in a))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines)+'\n')
    manifest=dict(checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha(checkpoint),config_sha256=sha(config_path) if config_path else None,
                  start=start,actor_arch='17-64-64-mu8-sigma8',tensor_order='row_major',native_model_sha256=sha(output),horizon=int(h),design=de,
                  rng='Native mt19937_64/normal_distribution is not NumPy PCG64 seed parity',source_config=source,effective_config=config)
    Path(str(output)+'.json').write_text(json.dumps(manifest,indent=2))
    return output

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--config',type=Path,help='Explicit evaluation_config.json or plain checkpoint-config object')
    ap.add_argument('--start',choices=('trained','initial_heads'),default='initial_heads')
    ap.add_argument('--horizon',type=int);ap.add_argument('--design',choices=('full_horizon','rollout_block'));args=ap.parse_args()
    print(export(args.checkpoint,args.output,args.start,args.config,args.horizon,args.design))
if __name__=='__main__':main()
