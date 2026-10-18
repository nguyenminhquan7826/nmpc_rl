"""Fast deterministic logic checks; no real NLP solve required."""
import numpy as np
import rl_nmpc_reg_v2A as rl

assert rl.REWARD_SCHEME.endswith('v2A')
assert tuple(rl.HI)==(20.,15.,50.)
assert np.isclose(rl.b.Ts,0.071)
assert np.isclose(rl.failure_cost_at_progress(0),3000)
assert np.isclose(rl.failure_cost_at_progress(.25),2500)
assert np.isclose(rl.failure_cost_at_progress(1),1000)

# The n-step target drops bootstrap on real failure but retains it on truncation.
rewards=np.array([-1.,-2501.]); future=np.array([-100.,-50.])
targets,horizons=rl.n_step_targets(rewards,future,[False,True],steps=2,gamma=.99)
assert np.isclose(targets[0],-1-.99*2501) and horizons[0]==2
assert np.isclose(targets[1],-2501) and horizons[1]==1
r2,h2=rl.n_step_targets(np.array([-1.]),np.array([-50.]),[False],steps=2,gamma=.99)
assert np.isclose(r2[0],-1-.99*50) and h2[0]==1

saved_pack=rl.pack_parameters
saved_build=rl.b.build_fresh_dynamics_rollout_guess
saved_fd=rl.b.Fd_plant
saved_unpack=rl.unpack_solution
saved_shift=rl.shift_warm_start
rl.pack_parameters=lambda *args,**kw: []
rl.b.build_fresh_dynamics_rollout_guess=lambda *args,**kw: None
rl.unpack_solution=lambda *args,**kw:(np.zeros((6,rl.b.N+1)),np.zeros((2,rl.b.N)))
rl.shift_warm_start=lambda *args,**kw:None

class FailedSolver:
    def __call__(self,**kw):
        raise RuntimeError('intentional solver failure')

class SuccessSolver:
    def __call__(self,**kw):
        return {'x':np.zeros(5), 'g':np.zeros(1)}
    def stats(self):
        return {'success':True}


def make_env(solver,progress=.25,limit=10):
    env=rl.Environment.__new__(rl.Environment)
    env.initial_weights=rl.initial_weights()
    env.problem={'solver':solver,'lbx':np.full(5,-1.),'ubx':np.full(5,1.),
                 'lbg':np.full(1,-1.),'ubg':np.full(1,1.)}
    env.limit=limit;env.k=0;env.fail=0;env.logs=[]
    env.x=np.zeros(6);env.prev=np.zeros(2);env.guess=None;env.rm=None;env.plant=np.array(rl.b.p_nominal)
    def observe():
        env.ref={'projection':{'s':progress*rl.b.CAD_S_END,'distance':.01},
                 'xref':np.zeros((rl.b.N+1,6))}
        return np.zeros(12),np.zeros(6)
    env.observe=observe
    return env

w=rl.initial_weights()
try:
    rl.b.Fd_plant=lambda x,u,p:np.array(x)
    # 3 failures: first two only local step cost, final adds progress cost.
    e=make_env(FailedSolver(),progress=.25)
    for i in range(3):
        _,_,reward,done,term=e.step(w)
        if i<2:
            assert not done and not term and reward==-10
        else:
            assert done and term and np.isclose(reward,-2510)
            assert e.logs[-1]['termination_reason']=='solver_fail_3x'
            assert np.isclose(e.logs[-1]['terminal_failure_cost'],2500)
    # Immediate successful route end has NO penalty.
    e=make_env(SuccessSolver(),progress=1.0)
    _,_,r,done,term=e.step(w)
    assert done and term and e.logs[-1]['reached_end']==1
    assert e.logs[-1]['terminal_failure_cost']==0.
    # Time limit with no failure: bootstrap truncation, no failure penalty.
    e=make_env(SuccessSolver(),progress=.5,limit=1)
    _,_,r,done,term=e.step(w)
    assert done and not term and e.logs[-1]['time_limit_truncated']==1
    assert e.logs[-1]['terminal_failure_cost']==0.
    # Invalid plant state: terminal failure, recovered last finite state for logging.
    rl.b.Fd_plant=lambda x,u,p:np.full(6,np.nan)
    e=make_env(SuccessSolver(),progress=.1)
    _,_,r,done,term=e.step(w)
    assert done and term and e.logs[-1]['termination_reason']=='invalid_state'
    assert np.isclose(e.logs[-1]['terminal_failure_cost'],2800)
finally:
    rl.pack_parameters=saved_pack
    rl.b.build_fresh_dynamics_rollout_guess=saved_build
    rl.b.Fd_plant=saved_fd
    rl.unpack_solution=saved_unpack
    rl.shift_warm_start=saved_shift
print('OK: 3 solver failures, successful completion, timeout truncation, invalid state, TD bootstrap.')
