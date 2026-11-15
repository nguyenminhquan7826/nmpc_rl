"""Prepare reviewable C++ inputs on the host; no training or serial activity.
Nominal low-speed hardware settings are explicit JSON, separate from the learned
weights and their original provenance. Route bytes are never modified.
"""
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def settings(c,route,learning):
 with route.open() as f:rows=list(csv.DictReader(f))
 if len(rows)<2:raise ValueError('Route requires two rows')
 speed=[float(r['vx_ref_mps']) for r in rows];progress=[float(r['s_m']) for r in rows]
 if not all(math.isfinite(x) for x in speed+progress):raise ValueError('Nonfinite route')
 if not all(c['min_reference_speed']<=v<=c['max_speed'] for v in speed):raise ValueError('Route speed outside hardware range; no automatic rescaling')
 if not all(b>a for a,b in zip(progress,progress[1:])):raise ValueError('Nonmonotonic route progress')
 limit=math.radians(c['max_steering_deg']);p=list(map(float,c['model']))
 if len(p)!=9 or not all(math.isfinite(v) and v>0 for v in p) or abs(sum(p[4:6])-.25)>1e-6:raise ValueError('Invalid hardware dynamics/wheelbase')
 if not(0<c['max_speed']<=.5 and 0<limit<=math.radians(22.835397) and c['servo_sign'] in [-1,1] and .04<=c['Ts']<=.1):raise ValueError('Invalid hardware limits')
 if not(0<c['solve_budget_ms']<1000*c['Ts'] and 40<=c['state_timeout_ms']<=150 and 0<c['ack_timeout_ms']<=150):raise ValueError('Invalid deadlines')
 q=learning['Qe_fixed'];r=learning['Re_fixed']
 if len(q)!=6 or len(r)!=2 or not all(math.isfinite(float(v)) and float(v)>0 for v in q+r):raise ValueError('Invalid reward weights')
 return dict(N=[c['N']],Ts=[c['Ts']],sqp_max_iter=[c['sqp_max_iter']],model=p,
 state_lower=[0,-math.inf,-math.inf,-c['yaw_rate_limit'],-math.inf,-math.inf],state_upper=[c['max_speed'],math.inf,math.inf,c['yaw_rate_limit'],math.inf,math.inf],
 input_lower=[c['acceleration_lower'],-limit],input_upper=[c['acceleration_upper'],limit],du_lower=c['du_lower'],du_upper=c['du_upper'],beta_max=[c['beta_max']],vy_abs_max=[c['vy_abs_max']],
 max_speed=[c['max_speed']],max_steering=[limit],servo_sign=[c['servo_sign']],task_q=q,task_r=r,feature_length=[progress[-1]-progress[0]],min_reference_speed=[c['min_reference_speed']],state_timeout_ms=[c['state_timeout_ms']],solve_budget_ms=[c['solve_budget_ms']],ack_timeout_ms=[c['ack_timeout_ms']])

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path);p.add_argument('--hardware',type=Path,default=ROOT/'deployment_cpp14/examples/hardware_nominal.json');p.add_argument('--start',choices=['initial_heads','trained'],default='initial_heads');p.add_argument('--horizon',type=int,default=20);a=p.parse_args()
 if a.output.exists():p.error('Use new output directory')
 import numpy as np
 sys.path.insert(0,str(Path(__file__).parent));from export_online_model import export
 with np.load(a.checkpoint,allow_pickle=False) as z:learning=json.loads(str(z['metadata'].item()))
 if a.config:
  extra=json.loads(a.config.read_text());learning.update(extra.get('checkpoint_config',extra))
 hardware=json.loads(a.hardware.read_text());values=settings(hardware,a.reference,learning)
 a.output.mkdir(parents=True);export(a.checkpoint,a.output/'initial.model',a.start,a.config,a.horizon,'full_horizon')
 with (a.output/'hardware.settings').open('w') as f:
  f.write('NMPC_HARDWARE_V1\n')
  for k,v in values.items():f.write(k+' '+str(len(v))+' '+' '.join(format(float(x),'.17g') for x in v)+'\n')
 (a.output/'reference.csv').write_bytes(a.reference.read_bytes());(a.output/'hardware_source.json').write_bytes(a.hardware.read_bytes())
 sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
 (a.output/'manifest.json').write_text(json.dumps(dict(checkpoint_sha256=sha(a.checkpoint),reference_sha256=sha(a.reference),hardware_json_sha256=sha(a.hardware),settings_sha256=sha(a.output/'hardware.settings'),native_model_sha256=sha(a.output/'initial.model'),vy_assumed_zero=True,kalman_used=False,control_mapping='v_request = clamp(vx+(a-model[6]*model[7])*Ts); inverse calibrated cubic servo; forward only',status='Prepared, not validated on hardware'),indent=2)+'\n')
 print(a.output)
if __name__=='__main__':main()
