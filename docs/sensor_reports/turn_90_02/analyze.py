from pathlib import Path
import os,json,csv,argparse
os.environ['MPLCONFIGDIR']='/tmp/turn90-02-mpl'
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('csv',help='Path to original turn_90_02.csv');args=parser.parse_args()
source=Path(args.csv)
d=pd.read_csv(source,dtype={'calib_hex':str});t=(d.imu_ms-d.imu_ms.iloc[0]).to_numpy()/1000
y=np.rad2deg(d.yaw_est_rad.to_numpy());r=np.rad2deg(d.r_measured_radps.to_numpy());g=np.r_[0,np.cumsum((r[1:]+r[:-1])*.5*np.diff(t))]
windows=[(2,10),(21,26),(33,36),(42,46),(52,56),(60.9,61.15)]
plateaus=[]
for lo,hi in windows:
 m=(t>=lo)&(t<=hi)
 plateaus.append({'start_s':lo,'end_s':hi,'median_time_s':float(np.median(t[m])),'heading_median_deg':float(np.median(y[m])),'gyro_integral_median_deg':float(np.median(g[m])),'gyro_mean_dps':float(r[m].mean()),'gyro_std_dps':float(r[m].std(ddof=1))})
turns=[]
for i,(a,b) in enumerate(zip(plateaus,plateaus[1:]),1):
 dy=b['heading_median_deg']-a['heading_median_deg'];dg=b['gyro_integral_median_deg']-a['gyro_integral_median_deg']
 turns.append({'turn':i,'direction_from_logged_sign':'left_positive' if dy>0 else 'right_negative','from_window_s':[a['start_s'],a['end_s']],'to_window_s':[b['start_s'],b['end_s']],'heading_change_deg':dy,'gyro_integral_change_deg':dg,'gyro_over_heading_ratio':dg/dy,'independent_reference_deg':None,'last_endpoint_short':i==5})
metrics={'samples':len(d),'accepted':int(d.accepted.sum()),'duration_STM32_s':float(t[-1]),'packet_sequence_gaps':int(np.maximum(np.diff(d.packet_seq)-1,0).sum()),'tx_interval_ms':[int(np.diff(d.tx_ms).min()),int(np.diff(d.tx_ms).max())],'imu_age_ms':[int((d.tx_ms-d.imu_ms).min()),int((d.tx_ms-d.imu_ms).max())],'max_sensor_skew_ms':int(abs(d.enc_ms-d.imu_ms).max()),'calibration_hex_counts':d.calib_hex.value_counts().to_dict(),'encoder_min_max':[int(d.enc_total.min()),int(d.enc_total.max())],'plateaus':plateaus,'turns':turns,'limitations':['Physical 90-degree reference not yet confirmed','No original UART log or host timestamps','Last heading plateau is only ~0.44 seconds','No scale or bias correction applied']}
(root/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
pd.DataFrame(turns).to_csv(root/'turn_comparison.csv',index=False)
fig,ax=plt.subplots(3,1,figsize=(12,10),sharex=True)
ax[0].plot(t,y,label='BNO relative heading');ax[0].plot(t,g,label='Integral of normalized gyro Z',alpha=.8);ax[0].set_ylabel('Angle (deg)');ax[0].legend()
ax[1].plot(t,r);ax[1].set_ylabel('Gyro Z (deg/s)')
ax[2].plot(t,y-g);ax[2].set_ylabel('Heading - integral (deg)');ax[2].set_xlabel('STM32 IMU elapsed time (s)')
for a in ax:
 a.grid(alpha=.25)
 for lo,hi in windows:a.axvspan(lo,hi,color='green',alpha=.09)
fig.suptitle('turn_90_02: green windows used for plateau comparisons; no external angle truth');fig.tight_layout();fig.savefig(root/'diagnostics.png',dpi=145)
fig,ax=plt.subplots(figsize=(10,4));x=np.arange(1,6);ax.bar(x-.18,[abs(v['heading_change_deg']) for v in turns],.36,label='Absolute heading change');ax.bar(x+.18,[abs(v['gyro_integral_change_deg']) for v in turns],.36,label='Absolute gyro integral');ax.axhline(90,color='gray',linestyle='--',label='Nominal 90 deg, not confirmed truth');ax.set(xlabel='Turn number (last endpoint short)',ylabel='deg',xticks=x);ax.legend();ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(root/'turn_angles.png',dpi=145)
print(json.dumps({'turns':turns},indent=2))
