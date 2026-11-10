"""Coupled native Actor/Critic -> runtime NMPC -> RK4 integration parity."""
import os,subprocess,tempfile,unittest
from pathlib import Path
import numpy as np
from test_online_ac import ROOT,model_fixture,write_values,read_values
from test_online_controller import reference_sequence
from test_fixed_nmpc import fixture
from nmpc_casadi import build_nmpc_solver,pack_parameters,unpack_solution,shift_warm_start
from gaussian_weight_actor import build_actor_observation
from actor_critic_updates import learn_one_transition
from nstep_actor import full_horizon_advantages,block_advantages,update_actor_block
from direct_p_critic import error_from_state
from rl_paper_core import paper_reward
EXE=Path(os.environ.get('FOUR_MODE_PROBE',str(ROOT/'build/four_mode_probe' if (ROOT/'build/four_mode_probe').is_file() else ROOT/'deployment_cpp14/build/four_mode_probe')))
MODES=('fixed','actor_adapt','critic_adapt','full_adapt')

def scenario(n=12):
    data,cfg,fd=fixture();x=np.asarray(data['state']);u=np.asarray(data['previous']);refs=[x.copy()];refs[0][5]=0
    for _ in range(n+10):refs.append(np.asarray(fd(refs[-1],u,data['model'])).ravel())
    refs=np.asarray(refs);meta=[]
    for i in range(n):meta.extend([i*.0071,0,(n+10)*.0071,.12,.12,refs[0,4],refs[0,5],refs[0,2],10])
    data.update(steps=[n],reference_sequence=refs.ravel(),feature_meta_sequence=meta,
                epsilon=np.random.default_rng(42).normal(0,.15,(n,8)).ravel(),task_q=[12,3,50,6,100,100],task_r=[1,1],terminal=[0])
    return data,cfg,fd

def python_rollout(actor,critic,model,data,cfg,fd,mode):
    n=int(data['steps'][0]);N=10;state=np.asarray(data['state']);previous=np.asarray(data['previous']);guess=data['guess'];refs=data['reference_sequence'].reshape(n+N+1,6);meta=np.asarray(data['feature_meta_sequence']).reshape(n,9);eps=np.asarray(data['epsilon']).reshape(n,8)
    solver=build_nmpc_solver(fd,N,np.diag(model['fixed_q']),np.diag(model['fixed_r']),np.diag(model['fixed_qf']),cfg,adaptive_cost_weights=True)
    ua=mode in ('actor_adapt','full_adapt');uc=mode in ('critic_adapt','full_adapt');h=int(model['horizon'][0]);pending=[];snapshot=None;out={};au=cu=samples=0
    config=dict(actor_lr=model['actor_lr'][0],actor_output_lr_factors=model['actor_factors'],actor_mu_reg_coef=model['mu_reg'][0],actor_mu_reg_threshold=model['mu_reg'][1])
    for i in range(n):
        m=meta[i];obs=build_actor_observation(state,refs[i],previous,path_progress_m=m[0],path_start_m=m[1],path_end_m=m[2],curvature_now=m[3],curvature_horizon_end=m[4],reference_origin_xy=m[5:7],reference_initial_heading_rad=m[7],train_path_length_m=m[8])
        mu,sigma=actor.forward(obs);latent=mu+sigma*eps[i] if ua else mu
        q,r=actor.physical_diagonals(latent) if mode!='fixed' else (model['fixed_q'],model['fixed_r'])
        p=critic.p if mode!='fixed' else model['fixed_qf'];err=error_from_state(state,refs[i]);
        par=pack_parameters(state,previous,refs[i:i+N+1],data['model'],weights=(q,r,p))
        sol=solver['solver'](x0=guess,p=par,lbx=solver['lbx'],ubx=solver['ubx'],lbg=solver['lbg'],ubg=solver['ubg']);assert solver['solver'].stats()['success']
        X,U=unpack_solution(np.asarray(sol['x']),N);control=U[:,0];nxt=np.asarray(fd(state,control,data['model'])).ravel();en=error_from_state(nxt,refs[i+1]);reward=float(paper_reward(err,control-previous,data['task_q'],data['task_r']))
        td=0.;batch=0;terminal=bool(data['terminal'][0]) and i==n-1
        if mode!='fixed':
            if not pending:snapshot=critic.p.copy()
            up=learn_one_transition(actor,critic,obs,latent,err,en,reward,gamma=model['gamma'][0],actor_lr=model['actor_lr'][0],critic_lr=1,critic_component_scales=model['critic_rates'],actor_output_scales=model['actor_factors'],actor_mu_reg_coef=model['mu_reg'][0],actor_mu_reg_threshold=model['mu_reg'][1],update_actor=ua and h==1,update_critic=uc,terminated=terminal)
            td=up.td;cu+=int(uc)
            if ua and h==1:au+=1;samples+=1;batch=1
            if ua and h>1:
                pending.append((obs.copy(),latent.copy(),err.copy(),reward))
                full=model['design'][0]==0
                if len(pending)>=(2*h-1 if full else h) or i==n-1:
                    rr=[x[3] for x in pending];ee=[x[2] for x in pending]
                    if full:a,_,_=full_horizon_advantages(rr,ee,en,snapshot,model['gamma'][0],terminal,h)
                    else:a,_=block_advantages(rr,ee,en,snapshot,model['gamma'][0],terminal)
                    if len(a):update_actor_block(actor,[x[0] for x in pending[:len(a)]],[x[1] for x in pending[:len(a)]],a,config);au+=1;samples+=len(a);batch=len(a)
                    pending=[]
        for k,v in [('states',state),('next_states',nxt),('controls',control),('weights',np.r_[q,r]),('p_before',p),('p_after',critic.p if mode!='fixed' else p),('reward',[reward]),('td',[td]),('batch_size',[batch])]:out.setdefault(k,[]).extend(v)
        state=nxt;previous=control;guess=shift_warm_start(X,U)
    out.update(actor_updates=[au],critic_updates=[cu],actor_samples=[samples])
    for k,v in actor.params.items():out['final_'+k]=v.ravel()
    return out

class RolloutTests(unittest.TestCase):
    def test_coupled_four_mode_parity(self):
        self.assertTrue(EXE.is_file(),'Native four-mode rollout runner missing')
        maxcontrol=maxweight=0.
        for mode,h,n in [(m,3,12) for m in MODES]+[('full_adapt',20,42)]:
            with self.subTest(mode=mode,horizon=h):
                actor,critic,model=model_fixture();model['horizon']=[h];data,cfg,fd=scenario(n)
                expected=python_rollout(actor,critic,model,data,cfg,fd,mode)
                with tempfile.TemporaryDirectory() as tmp:
                    tmp=Path(tmp);mp=tmp/'model';src=tmp/'case';dst=tmp/'out';write_values(mp,'AC_MODEL_V1',model);write_values(src,'AC_ROLLOUT_INPUT_V1',data)
                    run=subprocess.run([str(EXE),str(mp),str(src),str(dst),mode],text=True,capture_output=True);self.assertEqual(run.returncode,0,run.stderr)
                    out=read_values(dst,'AC_ROLLOUT_V1');self.assertTrue(Path(str(dst)+'.model').is_file())
                for k,v in expected.items():np.testing.assert_allclose(out[k],v,rtol=2e-7,atol=2e-8,err_msg=k)
                maxcontrol=max(maxcontrol,np.max(np.abs(out['controls']-expected['controls'])));maxweight=max(maxweight,np.max(np.abs(out['weights']-expected['weights'])))
        print('COUPLED controls_maxerr',maxcontrol,'weights_maxerr',maxweight)

    def test_native_seed_repeatability_and_failed_solve_no_learning(self):
        self.assertTrue(EXE.is_file(),'Native four-mode rollout runner missing')
        _,_,model=model_fixture();data,_,_=scenario(3);data.pop('epsilon')
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);mp=tmp/'model';src=tmp/'case';write_values(mp,'AC_MODEL_V1',model);write_values(src,'AC_ROLLOUT_INPUT_V1',data)
            results=[]
            for name,seed in [('a',2),('b',2),('c',1)]:
                dst=tmp/name;run=subprocess.run([str(EXE),str(mp),str(src),str(dst),'full_adapt',str(seed)],capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stderr);results.append(read_values(dst,'AC_ROLLOUT_V1'))
            np.testing.assert_array_equal(results[0]['weights'],results[1]['weights'])
            self.assertFalse(np.array_equal(results[0]['weights'],results[2]['weights']))
            data['state'][0]=.4;write_values(src,'AC_ROLLOUT_INPUT_V1',data);dst=tmp/'failed'
            run=subprocess.run([str(EXE),str(mp),str(src),str(dst),'full_adapt'],capture_output=True,text=True)
            self.assertEqual(run.returncode,2,run.stderr);out=read_values(dst,'AC_ROLLOUT_V1')
            self.assertEqual(out['complete'][0],0);self.assertEqual(out['actor_updates'][0],0);self.assertEqual(out['critic_updates'][0],0)
            self.assertNotIn('controls',out)
            final=read_values(Path(str(dst)+'.model'),'AC_MODEL_V1')
            for key in ('W1','W_mu','b_mu','critic_p'):np.testing.assert_array_equal(final[key],np.asarray(model[key]).ravel())

if __name__=='__main__':unittest.main()
