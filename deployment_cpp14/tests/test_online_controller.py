"""Independent parity for four mode update scheduling and model export."""
import copy,json,os,subprocess,tempfile,unittest
from pathlib import Path
import numpy as np
from test_online_ac import ROOT,PARAMS,model_fixture,write_values,read_values
from actor_critic_updates import learn_one_transition
from nstep_actor import block_advantages,full_horizon_advantages,update_actor_block
EXE=Path(os.environ.get('ONLINE_CONTROLLER_PROBE',str(ROOT/'deployment_cpp14/build/online_controller_probe')))
EXPORT=ROOT/'deployment_cpp14/tools/export_online_model.py'

def reference_sequence(actor,critic,d,mode,obs,eps,errors,en,rewards,terminal,horizon,design):
    ua=mode in (1,3);uc=mode in (2,3);pending=[];snapshot=None;updates=0;samples=0;rows={}
    cfg=dict(actor_lr=d['actor_lr'][0],actor_output_lr_factors=d['actor_factors'],actor_mu_reg_coef=d['mu_reg'][0],actor_mu_reg_threshold=d['mu_reg'][1])
    for i in range(len(rewards)):
        mu,sigma=actor.forward(obs[i]);z=mu+sigma*eps[i] if ua else mu
        w=np.r_[*actor.physical_diagonals(z)] if mode else np.r_[d['fixed_q'],d['fixed_r']]
        before=critic.p.copy() if mode else np.array(d['fixed_qf']);td=0.;batch=0
        rows.setdefault('weights',[]).extend(w);rows.setdefault('p_before',[]).extend(before)
        if mode:
            if not pending:snapshot=before.copy()
            up=learn_one_transition(actor,critic,obs[i],z,errors[i],en[i],rewards[i],gamma=d['gamma'][0],actor_lr=d['actor_lr'][0],critic_lr=1.,critic_component_scales=d['critic_rates'],actor_output_scales=d['actor_factors'],actor_mu_reg_coef=d['mu_reg'][0],actor_mu_reg_threshold=d['mu_reg'][1],critic_update_mode='residual_gradient' if d['critic_mode'][0] else 'semi_gradient',terminated=bool(terminal and i==len(rewards)-1),update_actor=ua and horizon==1,update_critic=uc)
            td=up.td
            if ua and horizon==1:updates+=1;samples+=1;batch=1
            if ua and horizon>1:
                pending.append((obs[i],z,errors[i],rewards[i]))
                collect=2*horizon-1 if design==0 else horizon
                if len(pending)>=collect or i==len(rewards)-1:
                    rr=np.array([x[3] for x in pending]);ee=np.array([x[2] for x in pending]);term=terminal and i==len(rewards)-1
                    if design==0:a,_,_=full_horizon_advantages(rr,ee,en[i],snapshot,d['gamma'][0],term,horizon)
                    else:a,_=block_advantages(rr,ee,en[i],snapshot,d['gamma'][0],term)
                    if len(a):
                        update_actor_block(actor,[x[0] for x in pending[:len(a)]],[x[1] for x in pending[:len(a)]],a,cfg)
                        updates+=1;samples+=len(a);batch=len(a)
                    pending=[]
        rows.setdefault('td',[]).append(td);rows.setdefault('batch_size',[]).append(batch)
        rows.setdefault('p_after',[]).extend(critic.p if mode else d['fixed_qf'])
    rows.update(actor_updates=[updates],actor_samples=[samples],critic_updates=[len(rewards) if uc else 0])
    for k,v in actor.params.items():rows['final_'+k]=v.ravel()
    return rows

class ControllerTests(unittest.TestCase):
    def test_four_modes_schedules_and_partial_flush(self):
        self.assertTrue(EXE.is_file(),'Native coordinator missing')
        largest=0.
        for mode,horizon,design,n,terminal in [(m,h,de,n,t) for m in range(4) for h,de,n,t in [(1,0,6,False),(20,0,45,False),(20,0,7,True),(20,0,7,False),(3,1,8,True)]]:
            with self.subTest(mode=mode,horizon=horizon,design=design,n=n,terminal=terminal):
                actor,critic,d=model_fixture();d.update(mode=[mode],horizon=[horizon],design=[design])
                rng=np.random.default_rng(52);obs=rng.normal(0,.3,(n,17));eps=rng.normal(size=(n,8));errors=rng.normal(0,.005,(n,6));en=rng.normal(0,.005,(n,6));rewards=-rng.uniform(.01,.03,n)
                d.update(observations=obs.ravel(),epsilon=eps.ravel(),errors=errors.ravel(),next_errors=en.ravel(),rewards=rewards,terminal=[int(terminal)])
                expected=reference_sequence(actor,critic,d,mode,obs,eps,errors,en,rewards,terminal,horizon,design)
                with tempfile.TemporaryDirectory() as tmp:
                    src=Path(tmp)/'in';dst=Path(tmp)/'out';write_values(src,'AC_SEQUENCE_V1',d)
                    run=subprocess.run([str(EXE),str(src),str(dst)],capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stderr)
                    out=read_values(dst)
                for k,v in expected.items():
                    np.testing.assert_allclose(out[k],v,rtol=1e-9,atol=2e-10,err_msg=k)
                    if k=='weights':largest=max(largest,float(np.max(np.abs(out[k]-v))))
        print('MODE SCHEDULE weights_maxerr',largest)

    def test_checkpoint_export_and_reset_heads(self):
        self.assertTrue(EXPORT.is_file(),'Checkpoint exporter missing')
        actor,critic,d=model_fixture()
        cfg=dict(Q_lower=d['lower'][:6].tolist(),Q_upper=d['upper'][:6].tolist(),R_lower=d['lower'][6:].tolist(),R_upper=d['upper'][6:].tolist(),P_lower=d['critic_lower'].tolist(),P_upper=d['critic_upper'].tolist(),initial_Q=d['fixed_q'],initial_R=d['fixed_r'],initial_P=d['fixed_qf'],fixed_Qf=d['fixed_qf'],sigma_lower=.05,sigma_initial=.2,sigma_upper=.6,gamma=.91,actor_lr=.0005,critic_lr=50000,critic_p_lr_factors=[.5,.5,.1,.05,4,4],actor_output_lr_factors=d['actor_factors'],train_length_m=10.)
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);ck=tmp/'checkpoint.npz'
            np.savez(ck,metadata=np.array(json.dumps(cfg)),critic_physical_P=critic.p,**{'actor_'+k:v for k,v in actor.params.items()})
            for start in ('trained','initial_heads'):
                dest=tmp/start
                run=subprocess.run([sys.executable,str(EXPORT),'--checkpoint',str(ck),'--output',str(dest),'--start',start],capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stderr);out=read_values(dest,'AC_MODEL_V1')
                np.testing.assert_array_equal(out['W1'],actor.params['W1'].ravel())
                np.testing.assert_array_equal(out['critic_p'],critic.p if start=='trained' else cfg['initial_P'])
                if start=='trained':np.testing.assert_array_equal(out['W_mu'],actor.params['W_mu'].ravel())
                else:
                    np.testing.assert_array_equal(out['W_mu'],np.zeros(512))
                    weights=out['lower']+.5*(out['upper']-out['lower'])*(1+np.tanh(out['b_mu']))
                    np.testing.assert_allclose(weights,np.r_[cfg['initial_Q'],cfg['initial_R']],atol=1e-12)
                self.assertEqual(json.loads(Path(str(dest)+'.json').read_text())['start'],start)

import sys
if __name__=='__main__':unittest.main()
