"""Offline spike: exercise the actual C++ yaw/bias KF on accepted CSV columns.
Run from anywhere with numpy, matplotlib and g++ installed. No firmware changes.
The two *_input.csv files are derived measurements, not original TEL packets.
"""
from pathlib import Path
import csv,io,json,subprocess,tempfile,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent
ROOT=D.parents[2]
CONFIGS=[(h,g,.01) for h in [.25,1,3] for g in [.1,.35,1]]+[(1,.35,.001),(1,.35,.1)]
DEFAULT=(1,.35,.01)
def key(c):return '/'.join(f'{v:g}' for v in c)
def run(exe,t,y,r,c):
 text=''.join(f'{a:.17g} {b:.17g} {d:.17g}\n' for a,b,d in zip(t,y,r))
 p=subprocess.run([str(exe),*map(str,c)],input=text,text=True,capture_output=True,check=True)
 return np.loadtxt(io.StringIO(p.stdout),delimiter=',').T

def deg(x):return np.rad2deg(x)
def rms(x):return float(np.sqrt(np.mean(np.square(x))))
def wrap(x):return np.arctan2(np.sin(x),np.cos(x))
def load(name):return np.loadtxt(D/(name+'_input.csv'),delimiter=',',skiprows=1).T

def main():
 results={'repository_commit_at_analysis':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), 'limitations':['CSV column replay of accepted samples, not original UART replay.','BNO NDOF heading and gyro are correlated; KF does not model that correlation.','No calibrated angle reference in hand-aligned turn data. Heading discrepancy is not true error or physical latency.','Source STM32 timestamps used unchanged; previous source/host clock ratio about 1.02 is not corrected.','SYS_ERR=9 recording excluded from parameter assessment. Turn log has no recorded fusion health fields.'], 'configs_units':'heading sigma deg / sampled gyro sigma deg/s / bias RW deg/s/sqrt(s)', 'cpp_sha256':hashlib.sha256((ROOT/'deployment_cpp14/src/yaw_kalman.cpp').read_bytes()).hexdigest()}
 with tempfile.TemporaryDirectory() as tmp:
  exe=Path(tmp)/'replay'
  subprocess.run(['g++','-std=c++14','-Wall','-Wextra','-Werror','-I'+str(ROOT/'deployment_cpp14/include'),str(D/'replay.cpp'),str(ROOT/'deployment_cpp14/src/yaw_kalman.cpp'),'-o',str(exe)],check=True)
  for name in ['stationary','turn']:
   t,y,r=load(name);assert np.all((np.diff(t)>0)&(np.diff(t)<=.2))
   window=t>=5; entries={};allruns={}
   for cfg in CONFIGS:
    ky,b=run(exe,t,y,r,cfg);allruns[cfg]=(ky,b)
    entry={'max_abs_discrepancy_to_BNO_heading_deg':float(np.max(np.abs(deg(wrap(ky[window]-y[window]))))), 'yaw_std_after_5s_deg':float(np.std(deg(ky[window]))),'discrepancy_to_BNO_heading_rms_deg':rms(deg(wrap(ky[window]-y[window]))),'bias_final_dps':float(deg(b[-1])),'bias_abs_max_dps':float(np.max(np.abs(deg(b))))}
    if name=='stationary':entry['yaw_peak_to_peak_after_5s_deg']=float(np.ptp(deg(ky[window])))
    entries[key(cfg)]=entry
   ky,b=allruns[DEFAULT]
   results[name]={'samples':len(t),'source_duration_s':float(t[-1]),'raw_yaw_std_after_5s_deg':float(np.std(deg(y[window]))),'raw_gyro_mean_after_5s_dps':float(np.mean(deg(r[window]))),'raw_gyro_std_after_5s_dps':float(np.std(deg(r[window]))),'configs':entries}
   if name=='turn':
    plateaus=[]
    for lo,hi in [(2,10),(21,26),(33,36),(42,46),(52,56)]:
     m=(t>=lo)&(t<=hi)
     plateaus.append({'window_s':[lo,hi],'raw_yaw_std_deg':float(np.std(deg(y[m]))),'kf_yaw_std_deg':float(np.std(deg(ky[m]))),'raw_heading_median_deg':float(np.median(deg(y[m]))),'kf_heading_median_deg':float(np.median(deg(ky[m]))),'median_bias_dps':float(np.median(deg(b[m])))})
    results[name]['default_plateaus']=plateaus
   with (D/(name+'_default_output.csv')).open('w') as f:
    w=csv.writer(f);w.writerow(['t_source_s','yaw_BNO_deg','yaw_KF_deg','gyro_dps','estimated_bias_dps','bias_corrected_rate_dps']);w.writerows(zip(t,deg(y),deg(ky),deg(r),deg(b),deg(r-b)))
   fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True)
   ax[0].plot(t,deg(y),label='BNO heading',lw=1.5);ax[0].plot(t,deg(ky),label='KF default',lw=1);ax[0].set_ylabel('Relative yaw (deg)');ax[0].legend()
   ax[1].plot(t,deg(wrap(ky-y)),lw=1);ax[1].set_ylabel('KF - BNO (deg)')
   ax[2].plot(t,deg(b),label='Estimated bias');ax[2].set_ylabel('Bias (deg/s)');ax[2].set_xlabel('STM32 source time (s)');ax[2].legend()
   for a in ax:a.grid(alpha=.3)
   fig.suptitle(name+': CSV replay of actual C++ KF (no angle truth)');fig.tight_layout();fig.savefig(D/(name+'.png'),dpi=150);plt.close(fig)
  # Known truth: independent white noises, constant gyro bias, wrapped heading.
  t=np.arange(0,60.0001,.02);y=.6*np.sin(.3*t)+.2*np.sin(.9*t);rate=.18*np.cos(.3*t)+.18*np.cos(.9*t)
  mask=t>=20;synthetic={key(c):[] for c in CONFIGS}
  for seed in range(10):
   rng=np.random.default_rng(seed);heading=wrap(y+np.deg2rad(rng.normal(0,.8,len(t))));gyro=rate+np.deg2rad(.4+rng.normal(0,.35,len(t)))
   for cfg in CONFIGS:
    ky,b=run(exe,t,heading,gyro,cfg)
    synthetic[key(cfg)].append({'seed':seed,'raw_yaw_rmse_deg':rms(deg(wrap(heading[mask]-y[mask]))),'kf_yaw_rmse_deg':rms(deg(wrap(ky[mask]-y[mask]))),'bias_final_dps':float(deg(b[-1]))})
    if seed==0 and cfg==DEFAULT:
     fig,ax=plt.subplots(2,1,figsize=(11,7),sharex=True)
     ax[0].plot(t,deg(wrap(heading-y)),label='Heading measurement error',alpha=.5,lw=.6);ax[0].plot(t,deg(wrap(ky-y)),label='KF error',lw=1);ax[0].set_ylabel('Yaw error (deg)');ax[0].legend()
     ax[1].axhline(.4,color='k',ls='--',label='True gyro bias');ax[1].plot(t,deg(b),label='Estimated bias');ax[1].set_ylabel('Bias (deg/s)');ax[1].set_xlabel('Synthetic time (s)');ax[1].legend()
     for a in ax:a.grid(alpha=.3)
     fig.suptitle('Independent-noise synthetic experiment, seed 0');fig.tight_layout();fig.savefig(D/'synthetic.png',dpi=150);plt.close(fig)
  results['synthetic']={'assumptions':'10 seeds 0..9, 60 s at 50Hz, constant bias 0.4 deg/s, independent sigma heading 0.8 deg and gyro 0.35 deg/s. Metrics after 20s. Not a BNO noise model.','configs':{k:{'mean_raw_rmse_deg':float(np.mean([r['raw_yaw_rmse_deg'] for r in v])),'mean_kf_rmse_deg':float(np.mean([r['kf_yaw_rmse_deg'] for r in v])),'worst_seed_kf_rmse_deg':max(r['kf_yaw_rmse_deg'] for r in v),'final_bias_range_dps':[min(r['bias_final_dps'] for r in v),max(r['bias_final_dps'] for r in v)]} for k,v in synthetic.items()}}
 (D/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
 with (D/'sensitivity.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['heading_sigma_deg','gyro_sigma_dps','bias_rw_dps','stationary_yaw_std_deg','turn_heading_discrepancy_rms_deg','turn_bias_final_dps','synthetic_mean_true_rmse_deg'])
  for c in CONFIGS:
   k=key(c);w.writerow([*c,results['stationary']['configs'][k]['yaw_std_after_5s_deg'],results['turn']['configs'][k]['discrepancy_to_BNO_heading_rms_deg'],results['turn']['configs'][k]['bias_final_dps'],results['synthetic']['configs'][k]['mean_kf_rmse_deg']])
 print(json.dumps({n:results[n]['configs'][key(DEFAULT)] for n in ['stationary','turn','synthetic']},indent=2))
if __name__=='__main__':main()
