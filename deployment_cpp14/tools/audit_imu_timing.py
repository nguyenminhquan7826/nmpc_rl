"""Offline yaw/gyro relative timing fit. Requires numpy; never shifts live data.
Positive lag means a later gyro value matches the earlier yaw change.
Only accepted rows from a continuous segment are supported.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np

def fit(t,yaw,rate,start=None,end=None):
    t=np.asarray(t);yaw=np.asarray(yaw);rate=np.asarray(rate)
    if len(t)<20 or not np.all(np.isfinite(np.r_[t,yaw,rate])) or np.any(np.diff(t)<=0) or np.max(np.diff(t))>.2:raise ValueError('Need finite continuous data with dt <=200ms')
    # Integrate gyro; compare average rates over centered ~100ms windows.
    g=np.r_[0,np.cumsum((rate[1:]+rate[:-1])*.5*np.diff(t))]
    w=5;a=t[:-w];b=t[w:];mid=(a+b)/2;dy=(yaw[w:]-yaw[:-w])/(b-a)
    keep=(a>=t[0]+.3)&(b<=t[-1]-.3)&(abs(dy)>np.deg2rad(3))
    if start is not None:keep&=mid>=start
    if end is not None:keep&=mid<=end
    a=a[keep];b=b[keep];dy=dy[keep]
    if len(dy)<20:raise ValueError('Not enough rotating samples')
    trials=[]
    for lag in np.arange(-.25,.2501,.005):
        avg=(np.interp(b+lag,t,g)-np.interp(a+lag,t,g))/(b-a)
        scale=float(avg@dy/(avg@avg))
        trials.append({'lag_ms':round(float(lag*1000),3),'rmse_dps':float(np.rad2deg(np.sqrt(np.mean((dy-avg)**2)))),'fitted_gain':scale,'gain_adjusted_rmse_dps':float(np.rad2deg(np.sqrt(np.mean((dy-scale*avg)**2))))})
    best=min(trials,key=lambda v:v['rmse_dps']);zero=min(trials,key=lambda v:abs(v['lag_ms']))
    return {'samples':len(dy),'best_unscaled':best,'zero_lag':zero,'search_boundary':abs(best['lag_ms'])>=250,'sign_convention':'positive: later gyro aligns with earlier heading change','curve':trials}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input');p.add_argument('--gyro-multiplier',type=int,choices=(-1,1),required=True);p.add_argument('--output',required=True);a=p.parse_args()
    rows=list(csv.DictReader(Path(a.input).open()))
    if any(r['accepted']!='1' for r in rows):raise ValueError('Segment rejected rows before analysis')
    ms=np.array([int(r['imu_ms']) for r in rows],dtype=np.int64)
    dt=((ms[1:]-ms[:-1])%2**32)/1000;t=np.r_[0,np.cumsum(dt)]
    yaw=np.array([float(r['yaw_est_rad']) for r in rows]);rate=np.array([float(r['r_measured_radps']) for r in rows])*a.gyro_multiplier
    results={'whole':fit(t,yaw,rate)}
    for lo,hi in [(16,22),(22,28),(28,30.56)]:
        try:results[f'{lo}-{hi}s']=fit(t,yaw,rate,lo,hi)
        except ValueError as e:results[f'{lo}-{hi}s']={'unavailable':str(e)}
    results['limitations']=['Relative signal alignment only, not physical sensor latency','BNO fusion and heading/gyro amplitude differences can confound lag','No live time shifting or gains applied']
    Path(a.output).write_text(json.dumps(results,indent=2))
    for k,v in results.items():
        if isinstance(v,dict):print(k,{key:val for key,val in v.items() if key!='curve'})
if __name__=='__main__':main()
