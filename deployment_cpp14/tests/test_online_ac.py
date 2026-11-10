"""Native online-learning parity against the unchanged Python implementation."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'nmpc_casadi'))
from gaussian_weight_actor import GaussianWeightActor,build_actor_observation
from direct_p_critic import SixParameterTDCritic
from nstep_actor import block_advantages,full_horizon_advantages
EXE=Path(os.environ.get('ACTOR_CRITIC_PROBE',str(ROOT/'deployment_cpp14/build/actor_critic_probe')))
PARAMS={'W1':(17,64),'b1':(64,),'W2':(64,64),'b2':(64,),
        'W_mu':(64,8),'b_mu':(8,),'W_sigma':(64,8),'b_sigma':(8,)}

def write_values(path,version,data):
    with open(path,'w') as f:
        f.write(version+'\n')
        for k,a in data.items():
            a=np.asarray(a).ravel();f.write(k+' '+str(a.size)+' '+' '.join(format(float(v),'.17g') for v in a)+'\n')

def read_values(path,version='AC_RESULT_V1'):
    t=path.read_text().split();assert t[0]==version
    result={};i=1
    while i<len(t):
        key=t[i];n=int(t[i+1]);result[key]=np.asarray(list(map(float,t[i+2:i+2+n])));i+=n+2
    return result

def model_fixture(seed=32):
    rng=np.random.default_rng(seed)
    low=np.array([1,.2,1,.5,10,10,.05,.5]);high=np.array([100,50,250,100,500,500,30,800])
    actor=GaussianWeightActor(rng,low[:6]+.4*(high[:6]-low[:6]),low[6:]+.4*(high[6:]-low[6:]),low[:6],high[:6],low[6:],high[6:],sigma_lower=.05,sigma_initial=.2,sigma_upper=.6)
    for k,shape in PARAMS.items():actor.params[k][...]=rng.normal(0,.12,shape)
    p=np.array([180,45,1800,90,2300,2280.]);plow=np.array([10,2,20,5,100,100]);phigh=np.array([1000,300,2000,500,25000,25000])
    critic=SixParameterTDCritic(p,plow,phigh)
    data={**{k:v.ravel().copy() for k,v in actor.params.items()},'lower':low,'upper':high,'sigma_bounds':[.05,.6],
          'critic_p':p,'critic_lower':plow,'critic_upper':phigh,'fixed_q':[12,3,120,6,153.75,152.5],'fixed_r':[15,100],
          'fixed_qf':[24,6,240,12,307.5,305], 'gamma':[.91],'actor_lr':[5e-4],
          'critic_rates':50000*np.array([.5,.5,.1,.05,4,4]),'actor_factors':[1,1,.5,.5,2,2,.5,.5],
          'mu_reg':[.03,.1],'critic_mode':[0],'horizon':[20],'design':[0],'train_length':[10.]}
    return actor,critic,data

class NativeMathTests(unittest.TestCase):
    def run_case(self,data,expected=0):
        self.assertTrue(EXE.is_file(),'Native Actor/Critic runner missing: '+str(EXE))
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'in';dst=Path(tmp)/'out';write_values(src,'AC_MATH_V1',data)
            run=subprocess.run([str(EXE),str(src),str(dst)],capture_output=True,text=True)
            self.assertEqual(run.returncode,expected,run.stdout+run.stderr)
            return read_values(dst) if dst.exists() else None

    def case(self,terminal=False,residual=False):
        actor,critic,data=model_fixture();rng=np.random.default_rng(55);n=7
        obs=rng.normal(0,.7,(n,17));epsilon=rng.normal(size=(n,8));mu,sigma=actor.forward(obs);latent=mu+sigma*epsilon
        adv=rng.normal(size=n);errors=rng.normal(0,.08,(n,6));en=rng.normal(0,.08,6);rewards=-rng.uniform(.2,1,n)
        data.update(observations=obs.ravel(),epsilon=epsilon.ravel(),latents=latent.ravel(),advantages=adv,
                    errors=errors.ravel(),next_error=en,rewards=rewards,error=errors[0],reward=[-2.],terminal=[int(terminal)],critic_mode=[int(residual)])
        state=np.array([.1,.02,np.pi+.01,.2,1,2]);ref=np.array([.11,0,-np.pi+.02,.1,.9,2.1]);previous=np.array([.2,.1])
        data.update(feature_state=state,feature_reference=ref,feature_previous=previous,feature_meta=[3,0,11,.2,.3,.1,.2,.3,10])
        return actor,critic,data,obs,latent,adv,errors,en,rewards

    def test_forward_all_gradients_and_projected_critic(self):
        for terminal,residual,reg in [(False,False,.03),(True,False,0.),(False,True,.03),(True,True,0.)]:
            with self.subTest(terminal=terminal,residual=residual,regularization=reg):
                actor,critic,d,obs,z,adv,errors,en,rewards=self.case(terminal,residual)
                d['mu_reg'][0]=reg;out=self.run_case(d)
                mu,sigma=actor.forward(obs);physical=np.asarray([np.r_[*actor.physical_diagonals(v)] for v in z])
                expected={'mu':mu,'sigma':sigma,'latent':z,'weights':physical,'logp':actor.latent_log_probability(obs,z)}
                loss,grad=actor.loss_and_gradients(obs,z,adv,output_gradient_scale=d['actor_factors'],mu_regularization_coef=reg,mu_regularization_threshold=.1)
                expected['loss']=[loss]
                for k,g in grad.items():expected['gradient_'+k]=g;expected['after_'+k]=actor.params[k]-d['actor_lr'][0]*g
                step=critic.td_step(errors[0],en,-2.,.91,d['critic_rates'],terminated=terminal,update_mode='residual_gradient' if residual else 'semi_gradient')
                expected.update(critic_td=[step.td_error],critic_gradient=step.gradient,critic_after=critic.p,critic_loss=[step.loss])
                a,t=block_advantages(rewards,errors,en,d['critic_p'],.91,terminal);expected.update(block_advantages=a,block_targets=t)
                a,t,h=full_horizon_advantages(rewards,errors,en,d['critic_p'],.91,terminal,3);expected.update(full_advantages=a,full_targets=t,full_horizons=h)
                meta=d['feature_meta'];expected['features']=build_actor_observation(d['feature_state'],d['feature_reference'],d['feature_previous'],path_progress_m=meta[0],path_start_m=meta[1],path_end_m=meta[2],curvature_now=meta[3],curvature_horizon_end=meta[4],reference_origin_xy=meta[5:7],reference_initial_heading_rad=meta[7],train_path_length_m=meta[8])
                for k,v in expected.items():np.testing.assert_allclose(out[k],np.asarray(v).ravel(),rtol=2e-10,atol=2e-10,err_msg=k)

    def test_invalid_model_or_transition_is_rejected(self):
        for key,value in [('W1',[0]),('lower',[-1]*8),('sigma_bounds',[.5,.1]),('actor_lr',[0]),('horizon',[1.5]),('critic_p',[np.nan]*6),('observations',[np.nan]*119),('terminal',[2])]:
            with self.subTest(key=key):
                _,_,d,*_=self.case();d[key]=value;self.assertIsNone(self.run_case(d,1))

if __name__=='__main__':unittest.main()
