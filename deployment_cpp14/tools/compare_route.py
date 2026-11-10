"""Full clean-route Python/C++ parity using original TrainEnvironment drivers.

C++ independently reads the original CSV and creates its own virtual tail,
projection and NMPC horizon. Shared epsilon is saved for repeatable comparison.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'nmpc_casadi'))
from export_online_model import export,SHAPES
from run_offline_trial import read_numbers,write_numbers
MODES=('fixed','actor_adapt','critic_adapt','full_adapt')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class EpsilonTape:
    def __init__(self,values):self.values=values;self.used=0
    def standard_normal(self,n):
        if n!=8 or self.used>=len(self.values):raise ValueError('Epsilon tape exhausted')
        value=self.values[self.used].copy();self.used+=1;return value

def numerical_check(actual,expected,atol,rtol):
    a=np.asarray(actual,float);b=np.asarray(expected,float)
    if a.shape!=b.shape:return dict(pass_=False,shape_cpp=list(a.shape),shape_python=list(b.shape))
    matches=np.isclose(a,b,atol=atol,rtol=rtol,equal_nan=True)
    delta=np.abs(np.subtract(a,b,where=np.isfinite(a)&np.isfinite(b),out=np.zeros_like(a)))
    bad=np.flatnonzero(~matches)
    return dict(pass_=bool(matches.all()),max_abs_error=float(np.max(delta)) if delta.size else 0.,mismatches=int(len(bad)),first_mismatch=int(bad[0]) if len(bad) else None)

class SolverCapture:
    """Replay failed Python NLP requests verbatim in the native solver probe."""
    def __init__(self,solver,env,settings,folder,binary,atol,rtol):
        self.solver=solver;self.env=env;self.settings=settings;self.folder=folder;self.binary=binary
        self.atol=atol;self.rtol=rtol;self.records=[]
    def __getattr__(self,name):return getattr(self.solver,name)
    def __call__(self,**kwargs):
        try:
            result=self.solver(**kwargs)
        except Exception:
            self.capture(kwargs)
            raise
        if not self.solver.stats().get('success',False):self.capture(kwargs)
        return result
    def capture(self,kwargs):
        if len(self.records)>=20:return
        self.folder.mkdir(exist_ok=True)
        params=np.asarray(kwargs['p'],float).ravel();n=int(self.settings['N'][0]);end=8+6*(n+1)
        keep=['N','Ts','sqp_max_iter','state_lower','state_upper','input_lower','input_upper','du_lower','du_upper','beta_max','vy_abs_max']
        case={k:self.settings[k] for k in keep}
        case.update(state=params[:6],previous=params[6:8],reference=params[8:end],model=params[end:end+9],q=params[end+9:end+15],r=params[end+15:end+17],qf=params[end+17:end+23],guess=np.asarray(kwargs['x0']).ravel(),dynamic_weights=[1])
        name=f'step_{self.env.k+1:04d}_failure_{len(self.records)+1:02d}'
        src=self.folder/(name+'.txt');dest=self.folder/(name+'.native');write_numbers(src,'FIXED_NMPC_V1',case)
        event=dict(step=self.env.k+1,python_success=False,input_file=src.name)
        if self.binary.is_file():
            with open(self.folder/(name+'.log'),'w') as log:run=subprocess.run([str(self.binary),str(src),str(dest)],stdout=log,stderr=subprocess.STDOUT)
            if run.returncode in (0,2):
                out=read_numbers(dest,'FIXED_NMPC_RESULT_V1');f,grad,g,jac=self.solver.get_function('nlp_jac_fg')(kwargs['x0'],kwargs['p'])
                checks={key:numerical_check(out[key],np.asarray(value).ravel(order='F'),self.atol,self.rtol) for key,value in [('eval_f',f),('eval_gradient',grad),('eval_g',g),('eval_jacobian',jac)]}
                event.update(cpp_success=bool(out['success'][0]),same_status=not bool(out['success'][0]),evaluation_checks=checks,evaluation_pass=all(v['pass_'] for v in checks.values()))
            else:event['replay_error']=run.returncode
        else:event['replay_error']='Build fixed_nmpc_probe to replay this saved input'
        self.records.append(event)

def construct_agents(model,config,mode,eps):
    from gaussian_weight_actor import GaussianWeightActor
    from direct_p_critic import SixParameterTDCritic
    if mode=='fixed':
        from run_experiment import FixedWeightActor
        return FixedWeightActor(model['fixed_q'],model['fixed_r']),SixParameterTDCritic(model['fixed_qf'],model['critic_lower'],model['critic_upper'])
    actor=GaussianWeightActor(np.random.default_rng(0),model['fixed_q'],model['fixed_r'],model['lower'][:6],model['upper'][:6],model['lower'][6:],model['upper'][6:],sigma_lower=model['sigma_bounds'][0],sigma_initial=config['sigma_initial'],sigma_upper=model['sigma_bounds'][1])
    for key,shape in SHAPES.items():actor.params[key][...]=model[key].reshape(shape)
    actor.rng=EpsilonTape(eps)
    return actor,SixParameterTDCritic(model['critic_p'],model['critic_lower'],model['critic_upper'])

def compare_one(native,python,final_model,actor,critic,model,summary,ns,mode,atol,rtol):
    checks={};columns=['X_m','Y_m','X_next_m','Y_next_m','X_ref_m','Y_ref_m','s_progress_m','progress_m','cross_track_m','e_psi','e_vx','a','steering','du_a','du_delta','reward','solver_success','solver_primary_success','solver_attempts','failure_streak','invalid_plant_state','reached_end','failure_terminal','terminated','truncated']
    for prefix,n in [('q',6),('r',2),('p',6),('obs',17),('state',6),('next_state',6)]:columns.extend(f'{prefix}_{i}' for i in range(n))
    if mode!='fixed':
        columns+=['td','training_reward','actor_param_step_norm','critic_param_step_norm','actor_batch_size','actor_update_applied','actor_advantage','actor_target','actor_advantage_horizon']
        for prefix,n in [('p_after',6),('mu',8),('sigma',8),('latent',8)]:columns.extend(f'{prefix}_{i}' for i in range(n))
    for col in columns:checks[col]=numerical_check(native[col],python[col],atol,rtol)
    for key in SHAPES:
        expected=model[key] if mode=='fixed' else actor.params[key].ravel()
        checks['final_'+key]=numerical_check(final_model[key],expected,atol,rtol)
    if mode!='fixed':checks['final_P']=numerical_check(final_model['critic_p'],critic.p,atol,rtol)
    for key in ['total_reward','discounted_cost_from_start','solver_success_pct','completion_pct','reached_end','failure_terminal','terminated','truncated','actor_updates','critic_updates','actor_training_samples']:
        if key in summary:checks['summary_'+key]=numerical_check(ns[key],[summary[key]],atol,rtol)
    passed=len(native)==len(python) and all(v['pass_'] for v in checks.values())
    trajectory_pass=len(native)==len(python) and all(v['pass_'] for k,v in checks.items() if k not in ('solver_primary_success','solver_attempts'))
    return {'trajectory_pass':bool(trajectory_pass),'pass':bool(passed),'steps_python':len(python),'steps_cpp':len(native),'reached_end':bool(ns['reached_end'][0]),'failure_terminal':bool(ns['failure_terminal'][0]),'checks':checks}

def plots(traces,reference,output,python_traces=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,7));ax.plot(reference.X_ref_m,reference.Y_ref_m,'k--',label='Reference')
    for mode,log in traces.items():ax.plot(np.r_[log.X_m,log.X_next_m.iloc[-1]],np.r_[log.Y_m,log.Y_next_m.iloc[-1]],label=mode+('' if log.reached_end.iloc[-1] else ' incomplete'))
    ax.axis('equal');ax.set(xlabel='X (m)',ylabel='Y (m)',title='Native C++14: CSV route');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(output/'trajectories.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));x=np.arange(4)
    for offset,key,label in [(-.2,'mean','Mean |CTE|'),(.2,'max','Max |CTE|')]:
        values=[100*getattr(traces[m].cross_track_m.abs(),key)() for m in MODES];bars=ax.bar(x+offset,values,.4,label=label);ax.bar_label(bars,fmt='%.2f')
    ax.set_xticks(x,[m+('' if traces[m].reached_end.iloc[-1] else '\nincomplete') for m in MODES],rotation=10);ax.set(ylabel='CTE (cm)',title='Complete logged rollout, including initial transient');ax.legend();fig.tight_layout();fig.savefig(output/'cte_mean_max.png',dpi=170);plt.close(fig)
    if python_traces is not None:
        fig,axes=plt.subplots(2,2,figsize=(11,10))
        for ax,mode in zip(axes.ravel(),MODES):
            ax.plot(reference.X_ref_m,reference.Y_ref_m,'k:',alpha=.4,label='Reference')
            for label,log,style in [('Python',python_traces[mode],'--'),('C++',traces[mode],'-')]:
                ax.plot(np.r_[log.X_m,log.X_next_m.iloc[-1]],np.r_[log.Y_m,log.Y_next_m.iloc[-1]],style,label=label)
            ax.set_title(mode);ax.axis('equal');ax.grid(alpha=.2);ax.legend()
        fig.tight_layout();fig.savefig(output/'python_cpp_overlay.png',dpi=170);plt.close(fig)
    log=traces['full_adapt'];fig,axes=plt.subplots(3,1,figsize=(10,10),sharex=True)
    for ax,prefix,names in [(axes[0],'q',['vx','vy','psi','r','X','Y']),(axes[1],'r',['acceleration','steering']),(axes[2],'p',['vx','vy','psi','r','X','Y'])]:
        for i,name in enumerate(names):ax.plot(log.s_progress_m,log[f'{prefix}_{i}'],label=name)
        ax.set_ylabel({'q':'Q','r':'R','p':'Qf = diag(P)'}[prefix]);ax.legend(ncol=3);ax.grid(alpha=.25)
    axes[-1].set_xlabel('Path progress (m)');fig.suptitle('Native Full adapt: applied weights');fig.tight_layout();fig.savefig(output/'full_adapt_weights.png',dpi=170);plt.close(fig)

def run(args):
    if args.output.exists():raise FileExistsError('Use a new output directory')
    if not args.binary.is_file():raise FileNotFoundError('Build route_controller using root CMake first')
    if not 1<=args.max_steps<=100000 or not 0<=args.seed<=2147483647 or not np.isfinite([args.ey,args.heading_deg,args.atol,args.rtol]).all() or min(args.atol,args.rtol)<0:raise ValueError('Invalid run parameters')
    args.output.mkdir(parents=True)
    model_path=args.output/'initial.model';export(args.checkpoint,model_path,args.start,args.config,args.horizon,args.design)
    manifest=json.loads(Path(str(model_path)+'.json').read_text());config=manifest['effective_config'].copy();model=read_numbers(model_path,'AC_MODEL_V1')
    reference_hash=digest(args.reference)
    # Bind learning settings to the actual exported artifact, including CLI schedule.
    config.update(gamma=float(model['gamma'][0]),actor_lr=float(model['actor_lr'][0]),critic_lr=1.,critic_p_lr_factors=model['critic_rates'].tolist(),actor_output_lr_factors=model['actor_factors'].tolist(),actor_mu_reg_coef=float(model['mu_reg'][0]),actor_mu_reg_threshold=float(model['mu_reg'][1]),critic_update_mode='residual_gradient' if model['critic_mode'][0] else 'semi_gradient',actor_advantage_steps=int(model['horizon'][0]),actor_advantage_design='rollout_block' if model['design'][0] else 'full_horizon')
    os.environ['NMPC_RL_MINIMAL_LAYOUT']='1'
    import baseline as b
    from cad_eval_reference import prepare_cad_evaluation_reference
    from train_rl_environment import TrainEnvironment
    from evaluate_frozen_actor_critic import frozen_rollout
    from online_adaptation import adaptive_rollout
    details,files=prepare_cad_evaluation_reference(args.reference,args.output/'python_reference',b.TAIL_LENGTH_M,min_speed_mps=float(b.cfg['vx_min']),max_speed_mps=min(float(b.TEST_SPEED_MPS),float(b.cfg['vx_max'])),wheelbase_m=float(b.WHEELBASE_M),steering_limit_rad=min(abs(float(b.cfg['u_min'][1])),abs(float(b.cfg['u_max'][1]))))
    if files['reference_sha256']!=reference_hash:raise RuntimeError('Reference changed while staging')
    b.REF_INFO=details;b.TAIL_FILE=Path(files['tail_file']);b.CAD_S_END=float(details['cad_s_end'])
    feature_length=b.CAD_S_END if args.feature_scale=='route' else float(model['train_length'][0])
    eps=np.random.default_rng(args.seed).standard_normal((args.max_steps,8));np.save(args.output/'epsilon.npy',eps)
    settings=dict(N=[b.N],Ts=[b.Ts],sqp_max_iter=[b.cfg['sqp_max_iter']],model=b.p_nominal,
                  state_lower=[b.cfg['vx_min'],-np.inf,-np.inf,-b.cfg['r_max'],-np.inf,-np.inf],state_upper=[b.cfg['vx_max'],np.inf,np.inf,b.cfg['r_max'],np.inf,np.inf],
                  input_lower=b.cfg['u_min'],input_upper=b.cfg['u_max'],du_lower=b.cfg['du_min'],du_upper=b.cfg['du_max'],beta_max=[b.cfg['beta_max']],vy_abs_max=[b.cfg['vy_abs_max']],
                  max_steps=[args.max_steps],seed=[args.seed],ey=[args.ey],heading_rad=[np.deg2rad(args.heading_deg)],tail_length=[b.TAIL_LENGTH_M],feature_length=[feature_length],min_reference_speed=[b.cfg['vx_min']],max_reference_speed=[min(b.TEST_SPEED_MPS,b.cfg['vx_max'])],
                  task_q=config['Qe_fixed'],task_r=config['Re_fixed'],solver_penalty=[config.get('train_solver_failure_penalty',0.)],failure_penalty=[config.get('train_failure_terminal_penalty',0.)],epsilon=eps.ravel())
    settings_path=args.output/'settings.txt';write_numbers(settings_path,'NMPC_ROUTE_V1',settings)
    reports={};traces={};python_traces={};metrics=[]
    for mode in MODES:
        print('Running Python/C++',mode,flush=True)
        actor,critic=construct_agents(model,config,mode,eps)
        env=TrainEnvironment(train_length_m=feature_length,q_initial=model['fixed_q'],r_initial=model['fixed_r'],p_initial=model['fixed_qf'] if mode=='fixed' else model['critic_p'],task_qe=np.asarray(config['Qe_fixed']),task_re=np.asarray(config['Re_fixed']),max_steps=args.max_steps,ey=args.ey,heading_deg=args.heading_deg)
        capture=SolverCapture(env.solver['solver'],env,settings,args.output/(mode+'_solver_replays'),args.binary.resolve().parent/'fixed_nmpc_probe',args.atol,args.rtol)
        env.solver['solver']=capture
        captured=[];observe=env.observe
        def recording_observe():
            obs=observe();captured.append(obs);return obs
        env.observe=recording_observe
        if mode=='fixed':summary,python=frozen_rollout(env,actor,critic,config['gamma'],mode);summary.update(actor_updates=0,critic_updates=0,actor_training_samples=0)
        else:summary,python=adaptive_rollout(env,actor,critic,config,update_actor=mode in ('actor_adapt','full_adapt'),update_critic=mode in ('critic_adapt','full_adapt'),sample_actions=mode in ('actor_adapt','full_adapt'))
        if len(captured)!=len(python)+1:raise RuntimeError('Unexpected Python observation lifecycle')
        augmented={}
        for prefix,values in [('obs',[o.features for o in captured[:-1]]),('state',[o.physical for o in captured[:-1]]),('next_state',[o.physical for o in captured[1:]])]:
            a=np.asarray(values)
            for i in range(a.shape[1]):augmented[f'{prefix}_{i}']=a[:,i]
        python=pd.concat([python,pd.DataFrame({k:v for k,v in augmented.items() if k not in python.columns})],axis=1)
        python.to_csv(args.output/(mode+'_python.csv'),index=False);python_traces[mode]=python
        if digest(args.reference)!=reference_hash:raise RuntimeError('Reference changed during evaluation')
        dest=args.output/(mode+'_cpp.csv');logpath=args.output/(mode+'_native_solver.log')
        with open(logpath,'w') as log:
            proc=subprocess.run([str(args.binary.resolve()),str(model_path),str(args.reference.resolve()),str(settings_path),str(dest),mode],stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode not in (0,2,3):raise RuntimeError('Native route run failed; see '+str(logpath))
        if digest(args.reference)!=reference_hash:raise RuntimeError('Reference changed during native run')
        native=pd.read_csv(dest);ns=read_numbers(Path(str(dest)+'.summary'),'ROUTE_SUMMARY_V1');final=read_numbers(Path(str(dest)+'.model'),'AC_MODEL_V1')
        report=compare_one(native,python,final,actor,critic,model,summary,ns,mode,args.atol,args.rtol);report['failed_solve_replays']=capture.records;reports[mode]=report;traces[mode]=native
        for language,frame in [('python',python),('cpp',native)]:metrics.append(dict(mode=mode,language=language,steps=len(frame),reached_end=bool(frame.reached_end.iloc[-1]),failure_terminal=bool(frame.failure_terminal.iloc[-1]),mean_cte_m=float(frame.cross_track_m.abs().mean()),max_cte_m=float(frame.cross_track_m.abs().max()),total_reward=float(frame.reward.sum())))
        print(mode,'PASS' if report['pass'] else ('TRAJECTORY_MATCH_RETRY_DIFFERENCE' if report['trajectory_pass'] else 'MISMATCH'),'steps',len(python),len(native),flush=True)
        (args.output/'parity.json').write_text(json.dumps(dict(all_pass=all(r['pass'] for r in reports.values()),all_trajectories_pass=all(r['trajectory_pass'] for r in reports.values()),modes=reports,atol=args.atol,rtol=args.rtol),indent=2))
    pd.DataFrame(metrics).to_csv(args.output/'summary.csv',index=False)
    plots(traces,pd.read_csv(args.reference),args.output,python_traces)
    run_manifest=dict(checkpoint_sha256=manifest['checkpoint_sha256'],reference_sha256=reference_hash,model_sha256=digest(model_path),settings_sha256=digest(settings_path),epsilon_sha256=digest(args.output/'epsilon.npy'),config_sha256=manifest['config_sha256'],
                      binary_sha256=digest(args.binary),seed=args.seed,start=args.start,feature_scale=args.feature_scale,feature_length=feature_length,legacy_wheelbase_m=b.WHEELBASE_M,Ts=b.Ts,N=b.N,clean_measurements=True,
                      algorithm='Original Python TrainEnvironment + adaptive_rollout/frozen_rollout vs independent native CSV route loop',native_fallback='two solves then hold previous, three consecutive failures terminate (Python simulation parity only)',effective_learning_config=config)
    (args.output/'manifest.json').write_text(json.dumps(run_manifest,indent=2))
    return all(r['pass'] for r in reports.values())

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--config',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--binary',type=Path,default=(ROOT/'build/route_controller' if (ROOT/'build/route_controller').is_file() else ROOT/'deployment_cpp14/build/route_controller'));ap.add_argument('--max-steps',type=int,default=2500);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--ey',type=float,default=.05);ap.add_argument('--heading-deg',type=float,default=0.)
    ap.add_argument('--start',choices=('trained','initial_heads'),default='initial_heads');ap.add_argument('--horizon',type=int,default=20);ap.add_argument('--design',choices=('full_horizon','rollout_block'),default='full_horizon');ap.add_argument('--feature-scale',choices=('route','train'),default='route')
    ap.add_argument('--atol',type=float,default=1e-7);ap.add_argument('--rtol',type=float,default=1e-6);args=ap.parse_args()
    raise SystemExit(0 if run(args) else 4)
if __name__=='__main__':main()
