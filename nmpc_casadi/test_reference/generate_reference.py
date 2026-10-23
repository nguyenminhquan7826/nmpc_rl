from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline,PchipInterpolator
from scipy.integrate import cumulative_trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent
w=pd.read_csv(root/'source_waypoints.csv')
u=w.s_m.to_numpy(float);x=w.x_m.to_numpy(float);y=w.y_m.to_numpy(float)
assert np.isfinite(np.c_[u,x,y]).all() and np.all(np.diff(u)>0)
sx=CubicSpline(u,x,bc_type='natural');sy=CubicSpline(u,y,bc_type='natural')
# Reparameterize spline by its actual numerical arc length.
ud=np.linspace(u[0],u[-1],max(20000,len(u)*50))
sd=cumulative_trapezoid(np.hypot(sx(ud,1),sy(ud,1)),ud,initial=0)
length=float(sd[-1]);s=np.r_[np.arange(0,length,.025),length]
q=PchipInterpolator(sd,ud)(s)
X=sx(q);Y=sy(q);dx=sx(q,1);dy=sy(q,1)
ddx=sx(q,2);ddy=sy(q,2)
psi=np.unwrap(np.arctan2(dy,dx))
kappa=(dx*ddy-dy*ddx)/(dx*dx+dy*dy)**1.5
V_MAX=.10;V_MIN=.05;AY_MAX=.01;A_MAX=.05;B_MAX=.05;EPS=1e-12
vlim=np.minimum(V_MAX,np.sqrt(AY_MAX/(np.abs(kappa)+EPS)))
if np.min(vlim)<V_MIN:
 raise ValueError('Curvature speed cap conflicts with simulator minimum; do not clip upward')
v=vlim.copy()
# Initial speed is maximum allowed at the starting point; no endpoint stop.
for i in range(len(v)-1):v[i+1]=min(v[i+1],np.sqrt(v[i]**2+2*A_MAX*(s[i+1]-s[i])))
for i in range(len(v)-2,-1,-1):v[i]=min(v[i],np.sqrt(v[i+1]**2+2*B_MAX*(s[i+1]-s[i])))
t=np.r_[0,np.cumsum(np.diff(s)/(.5*(v[:-1]+v[1:])))]
df=pd.DataFrame({'index':np.arange(len(s)),'s_m':s,'t_ref_s':t,
'vx_ref_mps':v,'vy_ref_mps':np.zeros(len(s)),'psi_ref_rad':psi,
'r_ref_radps':v*kappa,'X_ref_m':X,'Y_ref_m':Y,'kappa_1_per_m':kappa})
p=root/'test_xref.csv';df.to_csv(p,index=False,float_format='%.12g')
acc=np.diff(v*v)/(2*np.diff(s))
assert np.isfinite(df.to_numpy()).all()
assert np.max(v*v*np.abs(kappa))<=AY_MAX+1e-10
assert np.max(acc)<=A_MAX+1e-10 and np.min(acc)>=-B_MAX-1e-10
assert np.max(np.abs(np.arctan(.2515*kappa)))<=.55
assert np.min(v)>=V_MIN and np.max(v)<=V_MAX
info={'source':'reference_waypoints(1).csv','geometry':'natural cubic splines; numerical arc-length reparameterization',
'heading':'unwrap(atan2(dY/du,dX/du))',
'curvature':'(Xprime*Ysecond-Yprime*Xsecond)/(Xprime^2+Yprime^2)^1.5',
'speed_profile':'curvature cap followed by forward acceleration and backward braking passes',
'limits':{'v_max_mps':V_MAX,'v_min_compatible_mps':V_MIN,'ay_max_mps2':AY_MAX,'a_max_mps2':A_MAX,'braking_max_mps2':B_MAX},
'limit_provenance':'speed compatible with existing evaluator; acceleration limits are declared experimental generator settings, not measured vehicle limits',
'endpoint_stop':False,'sampling_m':.025,'points':len(s),'length_m':length,'duration_s':float(t[-1]),
'speed_min_mps':float(v.min()),'speed_max_mps':float(v.max()),'max_abs_curvature_1pm':float(abs(kappa).max()),
'max_abs_geometric_steering_rad':float(abs(np.arctan(.2515*kappa)).max()),
'longitudinal_accel_min_mps2':float(acc.min()),'longitudinal_accel_max_mps2':float(acc.max()),
'checks':'finite, increasing arc/time, yaw=v*kappa, lateral cap, longitudinal bounds, geometric steering',
'full_vehicle_simulation_verified':False,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(root/'reference_config.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
fig,ax=plt.subplots(2,2,figsize=(12,9),layout='constrained')
ax[0,0].plot(x,y,'k.',ms=2,label='Source waypoints');ax[0,0].plot(X,Y,label='Spline TEST')
ax[0,0].scatter(X[0],Y[0],label='Start');ax[0,0].scatter(X[-1],Y[-1],marker='x',label='End')
ax[0,0].set(xlabel='X (m)',ylabel='Y (m)');ax[0,0].axis('equal');ax[0,0].legend()
ax[0,1].plot(s,v);ax[0,1].set(xlabel='Arc length (m)',ylabel='Reference speed (m/s)')
ax[1,0].plot(s,kappa);ax[1,0].set(xlabel='Arc length (m)',ylabel='Curvature (1/m)')
ax[1,1].plot(s,v*kappa);ax[1,1].set(xlabel='Arc length (m)',ylabel='Reference yaw rate (rad/s)')
for a in ax.flat:a.grid(alpha=.25)
fig.suptitle('TEST reference from reference_waypoints(1).csv — no endpoint stop')
fig.savefig(root/'test_reference.png',dpi=180);fig.savefig(root/'test_reference.pdf');plt.close(fig)
print(json.dumps(info,indent=2))
