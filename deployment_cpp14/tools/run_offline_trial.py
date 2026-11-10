"""Run all four native C++ controllers on a short synthetic RK4 reference.

Python prepares files/plots only; four_mode_probe executes the control and
online learning loop. This is a port smoke test, not a real-route benchmark.
"""
import argparse
import csv
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'nmpc_casadi'))
from vehicle_model_casadi_variants import build_continuous_model,build_rk4_discrete_model
from export_online_model import export
MODES=('fixed','actor_adapt','critic_adapt','full_adapt')

def write_numbers(path,version,data):
    with open(path,'w') as f:
        f.write(version+'\n')
        for key,values in data.items():
            v=np.asarray(values).ravel();f.write(key+' '+str(len(v))+' '+' '.join(format(float(x),'.17g') for x in v)+'\n')

def read_numbers(path,version):
    t=path.read_text().split()
    if not t or t[0]!=version:raise ValueError('Wrong file version')
    out={};i=1
    while i<len(t):
        n=int(t[i+1]);out[t[i]]=np.array(list(map(float,t[i+2:i+2+n])));i+=n+2
    return out

def prepare_case(model,steps):
    ts=.071;N=10;L=.25;limit=np.deg2rad(22.835397)
    p=np.array([23.55/9.81,.044,.4,.35,L*132/252,L*120/252,.02,9.81,.01])
    fd=build_rk4_discrete_model(build_continuous_model(),ts)
    previous=np.array([.1962,0.]);reference=[np.array([.1,0,0,0,0,0.])]
    for i in range(steps+N):
        u=np.array([.1962,.04*np.sin(i/40.)]);reference.append(np.asarray(fd(reference[-1],u,p)).ravel())
    reference=np.asarray(reference);state=reference[0].copy();state[5]+=.01
    guess=[];x=state.copy()
    for _ in range(N):x=np.asarray(fd(x,previous,p)).ravel();guess.extend(x)
    meta=[]
    for i in range(steps):
        meta.extend([i*.1*ts,0,(steps+N)*.1*ts,float(reference[i,3]/reference[i,0]),float(reference[i+N,3]/reference[i+N,0]),0,0,0,model['train_length'][0]])
    return dict(N=[N],Ts=[ts],sqp_max_iter=[7],model=p,q=model['fixed_q'],r=model['fixed_r'],qf=model['fixed_qf'],state=state,previous=previous,
                state_lower=[.05,-np.inf,-np.inf,-4,-np.inf,-np.inf],state_upper=[.15,np.inf,np.inf,4,np.inf,np.inf],
                input_lower=[-2,-limit],input_upper=[2,limit],du_lower=[-.2,-.04],du_upper=[.2,.04],beta_max=[.6],vy_abs_max=[.6],
                reference=reference[:N+1].ravel(),guess=np.r_[guess,np.tile(previous,N)],steps=[steps],reference_sequence=reference.ravel(),feature_meta_sequence=meta,
                task_q=[12,3,50,6,100,100],task_r=[1,1],terminal=[0])

def summarize(mode,out,reference,folder):
    count=len(out.get('reward',[]));states=out.get('states',np.array([])).reshape(count,6)
    # Geometric distance to the supplied reference polyline, not odometry truth.
    xy=reference[:,4:6];a=xy[:-1];ab=np.diff(xy,axis=0);den=np.sum(ab*ab,axis=1);cte=[]
    for point in states[:,4:6]:
        t=np.clip(np.sum((point-a)*ab,axis=1)/np.maximum(den,1e-20),0,1)
        cte.append(float(np.min(np.linalg.norm(point-(a+t[:,None]*ab),axis=1))))
    with open(folder/(mode+'.csv'),'w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['step','vx','vy','psi','r','X','Y','a','delta','cte_m','reward','td','actor_batch_size']+['q_'+str(i) for i in range(6)]+['r_'+str(i) for i in range(2)]+['p_'+str(i) for i in range(6)])
        for i in range(count):writer.writerow([i,*states[i],*out['controls'][i*2:i*2+2],cte[i],out['reward'][i],out['td'][i],out['batch_size'][i],*out['weights'][i*8:i*8+8],*out['p_before'][i*6:i*6+6]])
    return dict(mode=mode,steps=count,complete=bool(out['complete'][0]),mean_cte_m=float(np.mean(cte)) if count else None,max_cte_m=max(cte) if count else None,
                total_reward=float(np.sum(out.get('reward',[]))),actor_updates=int(out['actor_updates'][0]),critic_updates=int(out['critic_updates'][0]))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--config',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=80);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--horizon',type=int,default=20)
    ap.add_argument('--start',choices=('initial_heads','trained'),default='initial_heads');ap.add_argument('--binary',type=Path,default=ROOT/'deployment_cpp14/build/four_mode_probe');args=ap.parse_args()
    if args.steps<1 or args.steps>10000 or not 0<=args.seed<2**64:ap.error('Invalid steps/seed')
    if not args.binary.is_file():ap.error('Build four_mode_probe first')
    if args.output.exists():ap.error('Use a new output directory')
    args.output.mkdir(parents=True)
    mp=args.output/'initial.model';export(args.checkpoint,mp,args.start,args.config,args.horizon,'full_horizon')
    model=read_numbers(mp,'AC_MODEL_V1');case=prepare_case(model,args.steps);src=args.output/'scenario.txt';write_numbers(src,'AC_ROLLOUT_INPUT_V1',case)
    traces={};records=[];ref=case['reference_sequence'].reshape(args.steps+11,6)
    for mode in MODES:
        dest=args.output/(mode+'.txt')
        run=subprocess.run([str(args.binary.resolve()),str(mp),str(src),str(dest),mode,str(args.seed)])
        if run.returncode not in (0,2):raise RuntimeError('Native runner failed: '+mode)
        out=read_numbers(dest,'AC_ROLLOUT_V1');traces[mode]=out;record=summarize(mode,out,ref,args.output);records.append(record);print(record)
    with open(args.output/'summary.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots();ax.plot(ref[:,4],ref[:,5],'k--',label='Synthetic reference')
    for mode,out in traces.items():
        state=out.get('states',np.array([])).reshape(-1,6);ax.plot(state[:,4],state[:,5],label=mode+('' if out['complete'][0] else ' incomplete'))
    ax.set(xlabel='X (m)',ylabel='Y (m)',title='Native C++14 offline smoke test');ax.axis('equal');ax.legend();fig.tight_layout();fig.savefig(args.output/'trajectories.png',dpi=160);plt.close(fig)
    (args.output/'README.txt').write_text('Synthetic short reference, native C++ RK4 plant, no sensor noise or UART. All modes start from independent copies. CTE is distance to the supplied polyline. Task Qe/Re are illustrative fixed smoke-test weights, not copied experiment rewards. This does not establish real-route learning superiority or Pi timing. Final .model files omit RNG and pending rollout buffers: weight warm starts only, not exact online resume.\n')
if __name__=='__main__':main()
