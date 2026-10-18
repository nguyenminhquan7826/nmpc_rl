"""NumPy Actor-Critic with n-step Critic and one-step Actor; no differentiation through NMPC.

REG-v2A (reward-only ablation against 5x32 v10): five 32-neuron tanh hidden layers.
Fixes the premature-failure reward loophole while preserving the unregularized
Actor/Critic networks, their update rules, initial Q/R/Qf, RL hyperparameters,
NMPC solver and fixed Q_EVAL/R_EVAL.  Timeout stays a bootstrap truncation.
The v6 value baseline is preserved: the critic value is
    V(s) = -b(s) - e' diag(F_BASE * m(s)) e ,   b(s) = B_SCALE * softplus(raw_6) >= 0
so the value can represent cost that is not explained by terminal-weight energy
(du^2 penalty, accumulated future tracking cost), while keeping V <= 0 like the reward.
Only the reward on a true failure termination is changed; the regularization remains OFF.
Also: sample-weighted update metrics, post-step progress logging, optimizer state in checkpoint,
CLI --sigma overrides checkpoint only when explicitly supplied.
Diagnostics distinguish quadratic energy from the full critic value.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import baseline as b
from nmpc_casadi import build_nmpc_solver, pack_parameters, unpack_solution, shift_warm_start

ROOT = Path(__file__).resolve().parent
# EDIT ONLY THESE THREE ARRAYS to choose the initial controller.
# State order: vx, vy, psi, r, X, Y; input increment order: a, delta.
Q_INIT = np.array([12., 3., 50., 6., 100.75, 100.75])
R_INIT = np.array([50., 10.])
QF_INIT = 20.0 * Q_INIT

# Keep reward weights unchanged when comparing different initial controllers.
Q_EVAL = np.array([20., 20., 100., 20., 600, 600])
R_EVAL = np.array([50, 100.])

# REG-v2A: terminal penalty ONLY for a true catastrophic episode termination.
# These are *initial experimental choices*, not theoretical guarantees.
# Keep unchanged when comparing with REG-v2B.
SOLVER_FAIL_STEP_COST = 10.0
FAIL_TERMINAL_BASE = 1000.0
FAIL_TERMINAL_REMAINING = 2000.0
REWARD_SCHEME = 'tracking_plus_progress_dependent_failure_penalty_v2A'

def failure_cost_at_progress(progress_fraction):
    """Penalty on TRUE task failure; success and timeout do not call this."""
    fraction = float(progress_fraction)
    if not np.isfinite(fraction):
        raise ValueError('Nonfinite route progress in terminal failure penalty')
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return FAIL_TERMINAL_BASE + FAIL_TERMINAL_REMAINING*(1.0-fraction)
Q_BASE = Q_INIT.copy()
R_BASE = R_INIT.copy()
F_BASE = QF_INIT.copy()
# Multipliers now relative to YOUR initial weights; initial multiplier = 1.
LO = np.array([0.1, 0.1, 0.1])
HI = np.array([20., 15., 50.])
FLO, FHI = 0.5, 50.0
# Non-negative value baseline b(s) = B_SCALE*softplus(raw); starts ~0.018 so initial V ~ old V.
B_SCALE = 1.0
B_INIT_RAW = -4.0

def initial_weights():
    for name, w, size in [('Q_INIT',Q_INIT,6),('R_INIT',R_INIT,2),('QF_INIT',QF_INIT,6)]:
        if w.shape != (size,) or not np.all(np.isfinite(w)) or np.any(w<=0):
            raise ValueError(f'{name} must have {size} positive finite diagonal entries')
    return Q_INIT.copy(), R_INIT.copy(), QF_INIT.copy()

SCALE = np.array([.1,.06,.2,.5,.1,.1,.1,2.,.55,2.,2.,1.])
GAMMA = .99
SIGMA = .05  # reduced for the wider action domain; configurable with --sigma

# Neural-network architecture study.
# Actor : 12 -> 32 -> 32 -> 32 -> 32 -> 32 -> 3
# Critic: 12 -> 32 -> 32 -> 32 -> 32 -> 32 -> 7
HIDDEN_SIZES = (32, 32, 32, 32, 32)
NN_ACTIVATION = 'tanh'
NN_INITIALIZER = 'xavier_uniform_hidden_zero_output'

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z,-40,40)))

def softplus(z):
    return np.logaddexp(0, z)

def n_step_targets(reward, next_values, terminated, steps, gamma=GAMMA):
    """Truncated n-step targets within a frozen rollout batch.

    True termination drops bootstrap. Batch/time truncation uses next_values
    at the last available transition, with the actual number of steps.
    """
    reward=np.asarray(reward,dtype=float)
    next_values=np.asarray(next_values,dtype=float)
    terminated=np.asarray(terminated,dtype=bool)
    if steps<1 or reward.ndim!=1 or reward.shape!=next_values.shape or reward.shape!=terminated.shape:
        raise ValueError('Invalid n-step target inputs')
    targets=np.zeros(len(reward));horizons=np.zeros(len(reward),dtype=int)
    for k in range(len(reward)):
        discount=1.
        for j in range(k,min(k+steps,len(reward))):
            targets[k]+=discount*reward[j]
            horizons[k]+=1
            discount*=gamma
            if terminated[j]:break
        if not terminated[j]:targets[k]+=discount*next_values[j]
    return targets,horizons


class MLP:
    """Fully-connected tanh MLP with five 32-neuron hidden layers."""
    def __init__(self, rng, outputs, bias, hidden_sizes=HIDDEN_SIZES):
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        if len(self.hidden_sizes) < 1 or any(x < 1 for x in self.hidden_sizes):
            raise ValueError('hidden_sizes must contain positive integers')

        dims = (12,) + self.hidden_sizes
        self.p = []

        # Hidden layers: Xavier-uniform initialization + zero bias.
        for nin, nout in zip(dims[:-1], dims[1:]):
            limit = np.sqrt(6.0 / (nin + nout))
            W = rng.uniform(-limit, limit, size=(nin, nout))
            b0 = np.zeros(nout)
            self.p.extend([W, b0])

        # Output weights start at zero so initial Q/R/Qf stay exactly as in v7.
        out_bias = np.asarray(bias, dtype=float)
        if out_bias.shape != (outputs,):
            raise ValueError(f'Output bias must have shape ({outputs},)')
        self.p.extend([
            np.zeros((dims[-1], outputs)),
            out_bias.copy(),
        ])

        self.m = [np.zeros_like(x) for x in self.p]
        self.v = [np.zeros_like(x) for x in self.p]
        self.t = 0

    @property
    def n_hidden(self):
        return len(self.hidden_sizes)

    def _hidden_forward(self, s):
        a = np.asarray(s, dtype=float)
        activations = [a]
        for layer in range(self.n_hidden):
            W = self.p[2*layer]
            b0 = self.p[2*layer + 1]
            a = np.tanh(a @ W + b0)
            activations.append(a)
        return activations

    def forward(self, s):
        activations = self._hidden_forward(s)
        h = activations[-1]
        return h @ self.p[-2] + self.p[-1], h

    def grads(self, s, dz):
        s = np.asarray(s, dtype=float)
        dz = np.asarray(dz, dtype=float)
        if s.ndim != 2 or dz.ndim != 2:
            raise ValueError('grads expects batched 2-D s and dz arrays')

        activations = self._hidden_forward(s)
        grads = [None] * len(self.p)

        # Output layer gradients.
        grads[-2] = activations[-1].T @ dz
        grads[-1] = dz.sum(axis=0)
        dh = dz @ self.p[-2].T

        # Hidden layers, backpropagated from layer 5 to layer 1.
        for layer in range(self.n_hidden - 1, -1, -1):
            h = activations[layer + 1]
            inp = activations[layer]
            da = dh * (1.0 - h*h)

            wi = 2*layer
            bi = wi + 1
            grads[wi] = inp.T @ da
            grads[bi] = da.sum(axis=0)

            if layer > 0:
                dh = da @ self.p[wi].T

        return grads

    def update(self, grads, lr):
        if len(grads) != len(self.p):
            raise ValueError('Gradient/parameter count mismatch')

        before = [x.copy() for x in self.p]
        norm = np.sqrt(sum(np.sum(g*g) for g in grads))
        factor = min(1., 1./max(norm, 1e-12))
        self.t += 1

        for i, g in enumerate(grads):
            g = g * factor
            self.m[i] = .9*self.m[i] + .1*g
            self.v[i] = .999*self.v[i] + .001*g*g
            self.p[i] -= lr * (
                self.m[i] / (1 - .9**self.t)
            ) / (
                np.sqrt(self.v[i] / (1 - .999**self.t)) + 1e-8
            )

        return {
            'grad_norm': float(norm),
            'clip_factor': float(factor),
            'parameter_delta': float(np.sqrt(
                sum(np.sum((x-y)**2) for x, y in zip(self.p, before))
            )),
        }

class Agent:
    def __init__(self, seed=0, sigma=SIGMA, actor_lr=1e-3, critic_lr=3e-3, critic_nsteps=16):
        self.critic_nsteps=int(critic_nsteps)
        if self.critic_nsteps<1 or self.critic_nsteps!=critic_nsteps:raise ValueError("critic_nsteps must be a positive integer")
        self.rng=np.random.default_rng(seed)
        self.actor_lr=float(actor_lr);self.critic_lr=float(critic_lr)
        if not np.all(np.isfinite([self.actor_lr,self.critic_lr])) or min(self.actor_lr,self.critic_lr)<=0: raise ValueError("learning rates must be positive")
        self.sigma=float(sigma)
        if not np.isfinite(self.sigma) or self.sigma<=0: raise ValueError('sigma must be positive')
        q,r,f=initial_weights()
        alpha=np.array([q[4]/Q_BASE[4],q[2]/Q_BASE[2],r[1]/R_BASE[1]])
        if not np.all(np.isfinite(alpha)) or np.any(HI <= LO) or np.any(alpha <= LO) or np.any(alpha >= HI):
            raise ValueError(f"Initial Actor multipliers must lie strictly inside bounds: alpha={alpha}, LO={LO}, HI={HI}. Adjust multiplier bounds explicitly.")
        unit=(alpha-LO)/(HI-LO)
        self.actor=MLP(self.rng,3,np.arctanh(2*unit-1))
        terminal=f/F_BASE
        if not np.all(np.isfinite(terminal)) or FHI <= FLO or np.any(terminal <= FLO) or np.any(terminal >= FHI):
            raise ValueError(f"Initial Critic multipliers must lie strictly inside ({FLO}, {FHI}): {terminal}")
        t=(terminal-FLO)/(FHI-FLO)
        # 7 outputs: 6 terminal-weight multipliers + 1 non-negative value baseline.
        self.critic=MLP(self.rng,7,np.concatenate([np.log(t/(1-t)),[B_INIT_RAW]]))
    def critic_terms(self,s):
        raw,_=self.critic.forward(s)
        sig=sigmoid(raw[...,:6])
        mult=FLO+(FHI-FLO)*sig
        base=B_SCALE*softplus(raw[...,6])
        return mult,sig,base,raw
    def propose(self,s,explore):
        mu,_=self.actor.forward(s)
        z=mu+(self.rng.normal(0,self.sigma,3) if explore else 0)
        a=LO+(HI-LO)*(np.tanh(z)+1)/2
        q=Q_BASE.copy(); q[2]*=a[1]; q[4:]*=a[0]
        r=R_BASE.copy(); r[1]*=a[2]
        mult,_,_,_=self.critic_terms(s)
        f=F_BASE*mult
        return (q,r,f),z
    def value(self,s,e):
        mult,_,base,_=self.critic_terms(s)
        return -base-np.sum(e*e*F_BASE*mult,axis=-1)
    def learn(self,rows):
        s,e,z,reward,sn,en,done=map(np.asarray,zip(*rows))
        mult,sig,base,raw=self.critic_terms(s)
        v=-base-np.sum(e*e*F_BASE*mult,axis=-1)
        # Both numeric targets are frozen before either network is updated.
        next_values=self.value(sn,en)
        actor_td=reward+GAMMA*(1-done)*next_values-v
        target,horizons=n_step_targets(reward,next_values,done,self.critic_nsteps)
        energy=np.sum(e*e*F_BASE,axis=1)
        # Range reachable by terminal-weight energy alone; b(s) is needed once target < vmin.
        vmin=-FHI*energy;vmax=-FLO*energy
        tol=1e-9*(1+np.abs(target))
        below=target<vmin-tol;above=target>vmax+tol
        td=target-v
        dv=np.empty((len(s),7))
        dv[:,:6]=-e*e*F_BASE*(FHI-FLO)*sig*(1-sig)
        dv[:,6]=-B_SCALE*sigmoid(raw[:,6])      # d(-b)/d raw = -B_SCALE*sigmoid(raw)
        near_upper=mult>=FHI-.01*(FHI-FLO)
        near_lower=mult<=FLO+.01*(FHI-FLO)
        gc=self.critic.update(self.critic.grads(s,(-td[:,None]*dv)/len(s)),self.critic_lr)
        mu,_=self.actor.forward(s)
        # Scale only; do not center away the sign of the estimated advantage.
        advantage=actor_td/max(float(np.std(actor_td)),1.)
        dz=-advantage[:,None]*(z-mu)/(self.sigma**2)/len(s)
        ga=self.actor.update(self.actor.grads(s,dz),self.actor_lr)
        logp=-.5*np.sum(((z-mu)/self.sigma)**2+2*np.log(self.sigma)+np.log(2*np.pi),axis=1)
        return {'critic_target_horizon_mean':float(horizons.mean()),
                'critic_target_horizon_min':int(horizons.min()),'critic_target_horizon_max':int(horizons.max()),
                'critic_full_nstep_pct':float(100*np.mean(horizons==self.critic_nsteps)),
                'actor_td_mean':float(actor_td.mean()),'actor_td_rms':float(np.sqrt(np.mean(actor_td**2))),
                'critic_loss':float(np.mean(td*td)/2),'actor_loss':float(np.mean(-advantage*logp)),
                'target_below_quadratic_min_pct':float(100*below.mean()),
                'target_above_quadratic_max_pct':float(100*above.mean()),
                'target_outside_quadratic_range_pct':float(100*(below|above).mean()),
                'quadratic_min_mean':float(vmin.mean()),'quadratic_max_mean':float(vmax.mean()),
                'target_above_full_value_upper_pct':float(100*above.mean()),
                'critic_near_upper_pct':float(100*near_upper.mean()),
                'critic_near_lower_pct':float(100*near_lower.mean()),
                'baseline_b_mean':float(base.mean()),'baseline_b_max':float(base.max()),
                'baseline_share_of_value_pct':float(100*base.mean()/max(float(np.mean(-v)),1e-12)),
                **{f'qf_multiplier_{i}_mean':float(mult[:,i].mean()) for i in range(6)},
                'td_mean':float(td.mean()),'td_rms':float(np.sqrt(np.mean(td*td))),
                'value_mean':float(v.mean()),'target_mean':float(target.mean()),
                **{'actor_'+k:x for k,x in ga.items()},**{'critic_'+k:x for k,x in gc.items()}}
    def _meta(self):
        return {'version':10,'reward_scheme':REWARD_SCHEME, 'solver_fail_step_cost':SOLVER_FAIL_STEP_COST, 'fail_terminal_base':FAIL_TERMINAL_BASE, 'fail_terminal_remaining':FAIL_TERMINAL_REMAINING, 'td_target':'n_step','critic_nsteps':self.critic_nsteps,'actor_advantage':'one_step_td','critic_baseline':[B_SCALE,B_INIT_RAW],
                'network_hidden_sizes':list(HIDDEN_SIZES),'network_activation':NN_ACTIVATION,
                'network_initializer':NN_INITIALIZER,
                'Q_INIT':Q_INIT.tolist(),'R_INIT':R_INIT.tolist(),'QF_INIT':QF_INIT.tolist(),
                'actor_lower':LO.tolist(),'actor_upper':HI.tolist(),'critic_bounds':[FLO,FHI],
                'Q_BASE':Q_BASE.tolist(),'R_BASE':R_BASE.tolist(),'Q_EVAL':Q_EVAL.tolist(),
                'R_EVAL':R_EVAL.tolist(),'gamma':GAMMA,'Ts':b.Ts,'N':b.N}
    def save(self,path):
        meta={**self._meta(),'sigma':self.sigma,'actor_lr':self.actor_lr,'critic_lr':self.critic_lr,'rng_state':self.rng.bit_generator.state}
        arrays={}
        for name in ['actor','critic']:
            net=getattr(self,name)
            arrays[f'{name}_param_count']=np.array(len(net.p))
            for i in range(len(net.p)):
                arrays[f'{name}{i}']=net.p[i]
                arrays[f'{name}_m{i}']=net.m[i]
                arrays[f'{name}_v{i}']=net.v[i]
            arrays[f'{name}_t']=np.array(net.t)
        np.savez(path, metadata=np.array(json.dumps(meta)), **arrays)
    def load(self,path,sigma_override=None):
        with np.load(path,allow_pickle=False) as d:
            if 'metadata' not in d: raise ValueError('Legacy checkpoint incompatible; train a fresh direct-weights model')
            meta=json.loads(str(d['metadata'].item()))
            for key,value in self._meta().items():
                if meta.get(key)!=value: raise ValueError('Checkpoint configuration mismatch: '+key)
            sigma = meta.get('sigma', SIGMA) if sigma_override is None else sigma_override
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError('Invalid checkpoint/override sigma')
            staged = {}
            for name in ['actor','critic']:
                net=getattr(self,name)
                count_key=f'{name}_param_count'
                if count_key not in d:
                    raise ValueError('Checkpoint network architecture metadata missing: '+name)
                param_count=int(d[count_key])
                if param_count != len(net.p):
                    raise ValueError(
                        f'Checkpoint network parameter count mismatch for {name}: '
                        f'{param_count} != {len(net.p)}'
                    )
                params=[d[f'{name}{i}'].copy() for i in range(param_count)]
                if any(x.shape!=y.shape or not np.all(np.isfinite(x)) for x,y in zip(params,net.p)):
                    raise ValueError('Invalid checkpoint network: '+name)
                keys=[f'{name}_t']+[f'{name}_{kind}{i}' for kind in ['m','v'] for i in range(param_count)]
                if not all(key in d for key in keys):
                    raise ValueError('Missing Adam state: '+name)
                m=[d[f'{name}_m{i}'].copy() for i in range(param_count)]
                v=[d[f'{name}_v{i}'].copy() for i in range(param_count)]
                t=d[f'{name}_t']
                if (any(x.shape!=y.shape or not np.all(np.isfinite(x)) for x,y in zip(m+v,params+params))
                    or any(np.any(x<0) for x in v) or t.shape!=() or not np.isfinite(t)
                    or float(t)<0 or float(t)!=int(t)):
                    raise ValueError('Invalid Adam state: '+name)
                staged[name]=(params,m,v,int(t))
            if 'rng_state' in meta:
                self.rng.bit_generator.state=meta['rng_state']
            for name,(params,m,v,t) in staged.items():
                net=getattr(self,name);net.p,net.m,net.v,net.t=params,m,v,t
            self.sigma=float(sigma)



class Environment:
    def __init__(self,max_seconds):
        self.initial_weights=initial_weights()
        self.limit=max(1,int(max_seconds/b.Ts))
        self.problem=build_nmpc_solver(b.Fd_controller,b.N,*[np.diag(w) for w in self.initial_weights],b.cfg,adaptive=True)
    def observe(self):
        self.ref=self.rm.get_nmpc_reference(self.x[4],self.x[5],b.Ts,b.N)
        xr=self.ref['xref'][0]; e=self.x-xr
        e[2]=np.arctan2(np.sin(e[2]),np.cos(e[2]))
        c,s=np.cos(xr[2]),np.sin(xr[2]); ep=c*e[4]+s*e[5]; ey=-s*e[4]+c*e[5]
        kh=self.rm.interpolate_curvature(self.ref['s_horizon'])
        obs=np.array([*e[:4],ep,ey,xr[0],*self.prev,kh[0],kh[-1],self.ref['projection']['s']/b.CAD_S_END])/SCALE
        return obs,e
    def reset(self,ey=.05,epsi=0,plant=None):
        self.rm=b.make_reference_manager()
        self.x,self.prev,self.guess,_=b.build_initial_state_and_guess(self.rm,self.problem,ey,epsi)
        self.plant=b.p_nominal.copy() if plant is None else plant.copy()
        self.k=0; self.fail=0; self.logs=[]
        return self.observe()
    def step(self,weights):
        old=self.x.copy(); obs,e=self.observe(); ref=self.ref
        progress_before=ref['projection']['s']
        p=pack_parameters(self.x,self.prev,ref['xref'],b.p_nominal,weights)
        start=time.perf_counter(); success=False; primary=False; violation=np.inf
        for attempt in range(2):
            guess=self.guess if attempt==0 else b.build_fresh_dynamics_rollout_guess(self.rm,self.x,self.prev,ref)
            try:
                sol=self.problem['solver'](x0=guess,p=p,lbx=self.problem['lbx'],ubx=self.problem['ubx'],lbg=self.problem['lbg'],ubg=self.problem['ubg'])
                z=np.asarray(sol['x']).ravel(); g=np.asarray(sol['g']).ravel()
                violation=max(float(np.max(np.maximum(self.problem['lbx']-z,0))),float(np.max(np.maximum(z-self.problem['ubx'],0))),float(np.max(np.maximum(self.problem['lbg']-g,0))),float(np.max(np.maximum(g-self.problem['ubg'],0))))
                success=bool(self.problem['solver'].stats().get('success',False)) and np.all(np.isfinite(z)) and violation<=1e-6
                if success:
                    primary=attempt==0; xs,us=unpack_solution(z,b.N); u=us[:,0].copy(); self.guess=shift_warm_start(xs,us); break
            except Exception:
                success=False
        elapsed=(time.perf_counter()-start)*1000
        if not success:u=self.prev.copy()
        self.fail=0 if success else self.fail+1
        du=u-self.prev

        # Preserved v10 per-step tracking objective and reference convention.
        state_cost=float(np.sum(Q_EVAL*e**2))
        control_cost=float(np.sum(R_EVAL*du**2))
        solver_fail_cost=SOLVER_FAIL_STEP_COST if not success else 0.0
        reward=-(state_cost+control_cost+solver_fail_cost)

        self.x=np.asarray(b.Fd_plant(self.x,u,self.plant)).ravel(); self.prev=u.copy(); self.k+=1
        invalid=not np.all(np.isfinite(self.x))
        if invalid:
            # Keep the last finite state so reference projection and diagnostics remain valid.
            self.x=old.copy()
        sn,en=self.observe()
        progress_after=float(self.ref['projection']['s'])
        progress_fraction=float(np.clip(progress_after/max(float(b.CAD_S_END),1e-12),0.0,1.0))

        # Important: failure takes precedence over reaching the endpoint.
        failure_terminal=bool(invalid or self.fail>=3)
        reached=bool(progress_after>=b.CAD_S_END-.03 and not failure_terminal)
        terminated=bool(reached or failure_terminal)

        # ONLY genuine failure terminations get a failure penalty.
        # A time cap remains a TRUNCATION and bootstraps V(next_state), as v10 did.
        terminal_failure_cost=(
            failure_cost_at_progress(progress_fraction) if failure_terminal else 0.0
        )
        reward-=terminal_failure_cost
        time_limit_truncated=bool(not terminated and self.k>=self.limit)
        done=bool(terminated or time_limit_truncated)

        termination_reason=(
            'invalid_state' if invalid else
            'solver_fail_3x' if self.fail>=3 else
            'reached_end' if reached else
            'time_limit_truncation' if time_limit_truncated else
            'running'
        )
        row={'t_s':(self.k-1)*b.Ts,'X':old[4],'Y':old[5],
             'X_ref':ref['xref'][0,4],'Y_ref':ref['xref'][0,5],
             'cte_m':ref['projection']['distance'],'e_psi':e[2],'e_vx':e[0],
             'reward':reward,'solver_success':int(success),
             'primary_success':int(primary),'constraint_violation':violation,
             'solve_ms':elapsed,'a':u[0],'delta':u[1],
             'du_a':du[0],'du_delta':du[1],
             'progress_before_m':progress_before,'progress_m':progress_after,
             'progress_fraction':progress_fraction,
             'state_cost':state_cost,'control_cost':control_cost,
             'solver_fail_cost':solver_fail_cost,
             'terminal_failure_cost':terminal_failure_cost,
             'failure_terminal':int(failure_terminal),
             'time_limit_truncated':int(time_limit_truncated),
             'termination_reason':termination_reason,
             'reached_end':int(reached),'terminated':int(terminated)}
        for name,w in zip(['q','r','qf'],weights):row.update({f'{name}_{i}':float(v) for i,v in enumerate(w)})
        self.logs.append(row)
        return sn,en,reward,done,terminated

def metrics(log):
    d=pd.DataFrame(log)
    result = {'return':float(d.reward.sum()),'mean_cte_m':float(d.cte_m.mean()),'max_cte_m':float(d.cte_m.max()),'rmse_heading_rad':float(np.sqrt(np.mean(d.e_psi**2))), 'rmse_vx_mps':float(np.sqrt(np.mean(d.e_vx**2))),'solver_success_pct':float(100*d.solver_success.mean()),'solve_p95_ms':float(d.solve_ms.quantile(.95)),'completion_pct':float(100*d.progress_m.iloc[-1]/b.CAD_S_END),'reached_end':bool(d.reached_end.iloc[-1]),'terminated':bool(d.terminated.iloc[-1]),'mean_du_delta_sq':float(np.mean(d.du_delta**2))}
    result.update(
        state_cost_sum=float(d.state_cost.sum()),
        control_cost_sum=float(d.control_cost.sum()),
        solver_fail_cost_sum=float(d.solver_fail_cost.sum()),
        terminal_failure_cost_sum=float(d.terminal_failure_cost.sum()),
        return_without_failure_penalty=float(d.reward.sum()+d.terminal_failure_cost.sum()),
        failure_terminal=bool(d.failure_terminal.iloc[-1]),
        time_limit_truncated=bool(d.time_limit_truncated.iloc[-1]),
        termination_reason=str(d.termination_reason.iloc[-1])
    )
    post=d[d.progress_before_m>=1.3]   # same sample selection as v5 (pre-step progress)
    result['post_sample_count']=len(post)
    result['mean_cte_post_m']=float(post.cte_m.mean()) if len(post) else np.nan
    result['max_cte_post_m']=float(post.cte_m.max()) if len(post) else np.nan
    result['rmse_heading_post_rad']=float(np.sqrt(np.mean(post.e_psi**2))) if len(post) else np.nan
    result['rmse_vx_post_mps']=float(np.sqrt(np.mean(post.e_vx**2))) if len(post) else np.nan
    result['tracking_feasible']=bool(result['reached_end'] and result['completion_pct']>=99
        and result['solver_success_pct']>=99 and len(post)>0
        and result['mean_cte_post_m']<=.05 and result['max_cte_post_m']<=.10
        and result['rmse_heading_post_rad']<=.20)
    return result


CONTROLLERS = ['baseline','fixed_qf','actor_fixed_qf','critic_only','actor_critic']


def controller_weights(agent_weights, initial, label):
    """Inference ablation; all learned outputs come from the same frozen checkpoint."""
    q,r,f=agent_weights
    qi,ri,fi=initial
    if label=='baseline': return qi.copy(),ri.copy(),fi.copy()
    if label=='fixed_qf': return qi.copy(),ri.copy(),FHI*F_BASE
    if label=='actor_fixed_qf': return q,r,FHI*F_BASE
    if label=='critic_only': return qi.copy(),ri.copy(),f
    if label=='actor_critic': return q,r,f
    raise ValueError('Unknown controller: '+str(label))


def selection_key(m, criterion):
    """Prefer feasible, then completed/solver-valid; compare only finite metrics."""
    healthy=bool(m['reached_end'] and m['completion_pct']>=99 and m['solver_success_pct']>=99)
    cte=m['mean_cte_post_m'];ret=m['return']
    cte=float(cte) if np.isfinite(cte) else float('inf')
    ret=float(ret) if np.isfinite(ret) else -float('inf')
    primary=(-cte,ret) if criterion=='tracking' else (ret,-cte)
    return (int(m['tracking_feasible']),int(healthy),*primary)



def rollout(env,agent,explore,ey,epsi,baseline=False,plant=None,train=False,rollout_steps=256,fixed_terminal=False,controller=None):
    s,e=env.reset(ey,epsi,plant); rows=[];updates=[];env.learning_updates=[]
    while True:
        weights,z=agent.propose(s,explore)
        label=controller or ('fixed_qf' if fixed_terminal else 'baseline' if baseline else 'actor_critic')
        if train and label!='actor_critic': raise ValueError('Ablation is evaluation only')
        weights=controller_weights(weights,env.initial_weights,label)
        value_before=float(agent.value(s,e)) if not train and label=='actor_critic' else None
        sn,en,r,done,term=env.step(weights)
        if value_before is not None:
            energy=float(np.sum(e*e*F_BASE))
            env.logs[-1].update(value=value_before,quadratic_min=-FHI*energy,
                quadratic_max=-FLO*energy,full_value_upper=-FLO*energy,next_value=float(agent.value(sn,en)),
                value_b=float(agent.critic_terms(s)[2]),
                curvature=float(s[9]*SCALE[9]),truncated=int(done and not term))
        rows.append((s.copy(),e.copy(),z.copy(),r,sn.copy(),en.copy(),float(term)))
        s,e=sn,en
        if train and (len(rows)>=rollout_steps or done):
            n=len(rows)
            update=agent.learn(rows);updates.append((n,update))
            env.learning_updates.append({'step':env.k,'batch_steps':n,**update});rows=[]
        if done:break
    m=metrics(env.logs)
    if updates:
        w=np.array([n for n,_ in updates],dtype=float)   # weight by number of samples in each batch
        for key in updates[0][1]:m[key]=float(np.average([v[key] for _,v in updates],weights=w))
        m['td_rms']=float(np.sqrt(np.average([v['td_rms']**2 for _,v in updates],weights=w)))
        m['actor_td_rms']=float(np.sqrt(np.average([v['actor_td_rms']**2 for _,v in updates],weights=w)))
        m['critic_target_horizon_min']=min(v['critic_target_horizon_min'] for _,v in updates)
        m['critic_target_horizon_max']=max(v['critic_target_horizon_max'] for _,v in updates)
        m['baseline_b_max']=max(v['baseline_b_max'] for _,v in updates)
        m['baseline_share_of_value_pct']=100*m['baseline_b_mean']/max(-m['value_mean'],1e-12)
        m['updates']=len(updates)
        m['td_rms_max']=max(v['td_rms'] for _,v in updates)
    return rows,m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['smoke','train','evaluate'],default='smoke')
    ap.add_argument('--actor-lr',type=float,default=1e-3)
    ap.add_argument('--critic-lr',type=float,default=3e-3)
    ap.add_argument('--rollout-steps',type=int,default=256)
    ap.add_argument('--critic-nsteps',type=int,default=16,help='Critic target length; 1 reproduces one-step TD, Actor keeps one-step TD')
    ap.add_argument('--eval-every',type=int,default=10)
    ap.add_argument('--sigma',type=float,default=None,help='Override exploration sigma; otherwise restore checkpoint sigma or use 0.05')
    ap.add_argument('--episodes',type=int,default=100)
    ap.add_argument('--seconds',type=float,default=120)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--output',type=Path,default=ROOT/'rl_results')
    args=ap.parse_args()
    if not np.isclose(b.Ts,0.071,atol=1e-10,rtol=0):
        ap.error(f'REG-v2A compares to v10 (Ts=0.071 s), but baseline.py has Ts={b.Ts}. Use the matching baseline.py.')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.seconds<=0 or not np.isfinite(args.seconds) or args.episodes<1:ap.error('seconds and episodes must be positive')
    if args.mode=='evaluate' and args.checkpoint is None:ap.error('evaluate requires --checkpoint')
    env=Environment(args.seconds); agent=Agent(args.seed,SIGMA if args.sigma is None else args.sigma,args.actor_lr,args.critic_lr,args.critic_nsteps)
    if args.checkpoint:agent.load(args.checkpoint,sigma_override=args.sigma)
    config={'td_target':'n_step','critic_nsteps':agent.critic_nsteps,'actor_advantage':'one_step_td','target_gradient':'stopped','critic_value':'-b(s)-e^T Qf(s) e, b>=0','network_hidden_sizes':list(HIDDEN_SIZES),'network_activation':NN_ACTIVATION,'network_initializer':NN_INITIALIZER,'actor_parameter_count':sum(x.size for x in agent.actor.p),'critic_parameter_count':sum(x.size for x in agent.critic.p),'b_scale':B_SCALE,'b_init_raw':B_INIT_RAW,'fixed_terminal_multiplier':FHI,'actor_lr':args.actor_lr,'critic_lr':args.critic_lr,'rollout_steps':args.rollout_steps,'eval_every':args.eval_every,'Ts':b.Ts,'N':b.N,'seed':args.seed,'seconds':args.seconds,'gamma':GAMMA,'sigma_fixed':agent.sigma,'actor_lower':LO.tolist(),'actor_upper':HI.tolist(),'critic_bounds':[FLO,FHI],'Q_INIT':env.initial_weights[0].tolist(),'R_INIT':env.initial_weights[1].tolist(),'QF_INIT':env.initial_weights[2].tolist(),'Q_EVAL':Q_EVAL.tolist(),'R_EVAL':R_EVAL.tolist(),'reward_weights_fixed':True,'checkpoint_source':str(args.checkpoint) if args.checkpoint else None,'evaluation_controllers':CONTROLLERS,'selection_priority':'feasible, completed with solver >=99%, then tracking or return on nominal validation'}
    config.update(
        experiment='REG-v2A_reward_only',
        regularization='none',
        reward_scheme=REWARD_SCHEME,
        solver_fail_step_cost=SOLVER_FAIL_STEP_COST,
        fail_terminal_base=FAIL_TERMINAL_BASE,
        fail_terminal_remaining=FAIL_TERMINAL_REMAINING,
        timeout_policy='truncate_and_bootstrap_no_terminal_failure_cost',
        expected_baseline_Ts=0.071,
    )
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    if args.mode in ['train','smoke']:
        history=[];validation=[];update_history=[];total_updates=0;best_keys={};best_records={}
        if args.rollout_steps<1 or args.eval_every<1:ap.error("rollout/eval interval must be positive")
        count=args.episodes if args.mode=='train' else 2
        for ep in range(count):
            ey=.05 if args.mode=='smoke' else agent.rng.uniform(-.05,.05)
            heading=0 if args.mode=='smoke' else agent.rng.uniform(-np.deg2rad(5),np.deg2rad(5))
            rows,m=rollout(env,agent,True,ey,heading,train=True,rollout_steps=args.rollout_steps)
            update_history.extend({'episode':ep,'update_index':total_updates+i+1,**v} for i,v in enumerate(env.learning_updates))
            total_updates+=m['updates']
            pd.DataFrame(update_history).to_csv(args.output/'update_history.csv',index=False)
            loss={"total_updates":total_updates}; history.append({'episode':ep,'initial_ey':ey,'initial_epsi':heading,**m,**loss})
            if not all(np.all(np.isfinite(p)) for net in [agent.actor,agent.critic] for p in net.p):raise RuntimeError('Nonfinite network parameters')
            pd.DataFrame(env.logs).to_csv(args.output/f'episode_{ep:04d}.csv',index=False)
            pd.DataFrame(history).to_csv(args.output/'training_history.csv',index=False)
            agent.save(args.output/'checkpoint.npz')
            print(json.dumps(history[-1]),flush=True)
            if ep==0 or (ep+1)%args.eval_every==0 or ep==count-1:
                for label in CONTROLLERS:
                    _,vm=rollout(env,agent,False,.05,0,controller=label)
                    validation.append({'episode':ep,'total_updates':total_updates,'controller':label,**vm})
                    if label=='actor_critic':
                        for criterion in ['tracking','return']:
                            key=selection_key(vm,criterion)
                            if criterion not in best_keys or key>best_keys[criterion]:
                                best_keys[criterion]=key
                                checkpoint_name=f'checkpoint_best_{criterion}.npz'
                                agent.save(args.output/checkpoint_name)
                                best_records[criterion]={'episode':ep,'total_updates':total_updates,
                                    'checkpoint':checkpoint_name,'selection_key':list(key),'metrics':vm}
                        (args.output/'best_checkpoints.json').write_text(json.dumps(best_records,indent=2))
                pd.DataFrame(validation).to_csv(args.output/'validation_history.csv',index=False)
        from plot_training_weights import plot_training_weights
        plot_training_weights(args.output)
    if args.mode=='evaluate' and args.checkpoint is None:ap.error('evaluate requires --checkpoint')
    scenarios=[('nominal',.05,0,b.p_nominal.copy())]
    if args.mode=='evaluate':
        for ey,deg in [(-.05,5),(.05,-5),(.10,-10)]:scenarios.append((f'ey{ey}_psi{deg}',ey,np.deg2rad(deg),b.p_nominal.copy()))
        for index,name in [(0,'mass'),(1,'Iz')]:
            for factor in [.9,1.1]:
                p=b.p_nominal.copy();p[index]*=factor;scenarios.append((f'{name}_{factor}',.05,0,p))
    results=[]
    for name,ey,heading,plant in scenarios:
        for label in CONTROLLERS:
            _,m=rollout(env,agent,False,ey,heading,plant=plant,controller=label)
            if label=='actor_critic':
                from critic_diagnostics import export_diagnostics
                diagnostic=export_diagnostics(env.logs,args.output,name,GAMMA)
                m.update(diagnostic)
            pd.DataFrame(env.logs).to_csv(args.output/f'{name}_{label}.csv',index=False)
            results.append({'scenario':name,'controller':label,**m})
    pd.DataFrame(results).to_csv(args.output/'comparison.csv',index=False)
    comparison=pd.DataFrame(results)
    paired=[]
    for scenario,group in comparison.groupby('scenario',sort=False):
        group=group.set_index('controller')
        for lhs,rhs,question in [
            ('actor_fixed_qf','fixed_qf','Actor effect with Qf fixed'),
            ('actor_critic','critic_only','Actor effect with adaptive Qf'),
            ('critic_only','fixed_qf','Adaptive Qf effect with initial Q/R'),
            ('actor_critic','actor_fixed_qf','Adaptive Qf effect with learned Q/R')]:
            row={'scenario':scenario,'comparison':question,'left_controller':lhs,'right_controller':rhs}
            for metric in ['return','mean_cte_post_m','max_cte_post_m','rmse_heading_post_rad','rmse_vx_post_mps']:
                row[metric+'_delta']=float(group.loc[lhs,metric]-group.loc[rhs,metric])
            paired.append(row)
    pd.DataFrame(paired).to_csv(args.output/'ablation_deltas.csv',index=False)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for label in CONTROLLERS:
        d=pd.read_csv(args.output/f'nominal_{label}.csv')
        axes[0].plot(d.t_s,d.cte_m*100,label=label)
        axes[1].plot(d.t_s,d.e_vx,label=label)
    axes[0].set_ylabel('CTE [cm]');axes[1].set_ylabel('vx error [m/s]');axes[1].set_xlabel('Time [s]')
    for ax in axes:ax.legend(ncol=3);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.output/'ablation_tracking.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots()
    for label in CONTROLLERS:
        d=pd.read_csv(args.output/f'nominal_{label}.csv');ax.plot(d.X,d.Y,label=label)
    ax.plot(d.X_ref,d.Y_ref,'k--',label='reference');ax.axis('equal');ax.legend();ax.set(xlabel='X [m]',ylabel='Y [m]');fig.savefig(args.output/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
    adaptive=pd.read_csv(args.output/'nominal_actor_critic.csv')
    for ax,prefix in zip(axes,['q','r','qf']):
        for col in adaptive.columns:
            if col.startswith(prefix+'_'):ax.plot(adaptive.t_s,adaptive[col],label=col)
        ax.set_ylabel(prefix);ax.legend(ncol=3)
    axes[-1].set_xlabel('time [s]');fig.tight_layout();fig.savefig(args.output/'adaptive_weights.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
    for col,base in [('q_4',Q_BASE[4]),('q_2',Q_BASE[2]),('r_1',R_BASE[1])]:
        axes[0].plot(adaptive.t_s,adaptive[col]/base,label=col+'/base')
    for i in range(6): axes[1].plot(adaptive.t_s,adaptive[f'qf_{i}']/F_BASE[i],label=f'qf_{i}/base')
    for i in range(6): axes[2].plot(adaptive.t_s,adaptive[f'qf_{i}']/env.initial_weights[2][i],label=f'qf_{i}/init')
    for ax,label in zip(axes,['Actor / base','Critic / base','Critic / init']):
        ax.set_ylabel(label);ax.legend(ncol=3)
    axes[-1].set_xlabel('time [s]');fig.tight_layout();fig.savefig(args.output/'weight_multipliers.png',dpi=160);plt.close(fig)
    print(pd.DataFrame(results).to_string(index=False))

if __name__=='__main__':main()