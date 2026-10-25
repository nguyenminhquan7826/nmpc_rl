"""Sensor-only diagnostic using analytic motion; not a controller evaluation."""
import argparse,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from sensor_models import SensorConfig,VehicleSensors,wrap_angle


def simulate(config,seed,duration):
    sensor=VehicleSensors(config,seed);rows=[];dt=config.sample_period_s
    # Analytic travel: smooth start to ~0.1 m/s and a slow speed variation.
    def distance(t):return .1*(t-.3*(1-np.exp(-t/.3)))+.01/.4*(1-np.cos(.4*t))
    for k in range(int(duration/dt)):
        t=(k+1)*dt;s=distance(t);delta=s-distance(t-dt)
        yaw=.3*np.sin(.25*t);rate=.075*np.cos(.25*t)
        row=sensor.sample(delta,yaw,rate)
        row.update(vx_interval_true_mps=delta/dt,distance_true_m=s,yaw_true_rad=yaw,yaw_rate_true_rad_s=rate)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--duration',type=float,default=30.)
    ap.add_argument('--yaw-noise-deg',type=float,default=.5)
    ap.add_argument('--rate-noise',type=float,default=.01)
    ap.add_argument('--filter-tau',type=float,default=.05)
    args=ap.parse_args()
    if not np.isfinite(args.duration) or args.duration<.01:ap.error('duration must be >= 0.01 s')
    if args.output.exists() and any(args.output.iterdir()):ap.error('Output must be empty')
    c=SensorConfig(yaw_noise_std_rad=np.deg2rad(args.yaw_noise_deg),yaw_rate_noise_std_rad_s=args.rate_noise,velocity_filter_tau_s=args.filter_tau)
    d=simulate(c,args.seed,args.duration)
    args.output.mkdir(parents=True,exist_ok=True);d.to_csv(args.output/'sensor_log.csv',index=False)
    summary=dict(role='analytic sensor diagnostic, no controller or position estimator',seed=args.seed,
        imu_settings_role='illustrative assumptions, not measured BNO055 specifications',config=asdict(c),
        counts_per_wheel_rev=c.counts_per_wheel_rev,metres_per_count=c.metres_per_count,
        speed_quantization_mps=c.metres_per_count/c.sample_period_s,
        raw_speed_rmse_mps=float(np.sqrt(np.mean((d.vx_raw_mps-d.vx_interval_true_mps)**2))),
        filtered_speed_rmse_mps=float(np.sqrt(np.mean((d.vx_filtered_mps-d.vx_interval_true_mps)**2))),
        yaw_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(wrap_angle(d.yaw_measured_rad-d.yaw_true_rad)**2)))),
        yaw_rate_rmse_rad_s=float(np.sqrt(np.mean((d.yaw_rate_measured_rad_s-d.yaw_rate_true_rad_s)**2))))
    (args.output/'sensor_config.json').write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(11,12),sharex=True)
    for key,label in [('vx_interval_true_mps','True interval speed'),('vx_raw_mps','Encoder raw'),('vx_filtered_mps','Encoder filtered')]:axes[0].plot(d.t_s,d[key],label=label,alpha=.8)
    axes[0].set_ylabel('Speed (m/s)');axes[0].legend()
    axes[1].plot(d.t_s,np.rad2deg(wrap_angle(d.yaw_measured_rad-d.yaw_true_rad)));axes[1].set_ylabel('Yaw error (deg)')
    axes[2].plot(d.t_s,d.yaw_rate_measured_rad_s-d.yaw_rate_true_rad_s);axes[2].set_ylabel('Yaw-rate error (rad/s)')
    axes[3].plot(d.t_s,1000*(d.encoder_distance_m-d.distance_true_m));axes[3].set_ylabel('Count travel error (mm)');axes[3].set_xlabel('Time (s)')
    for ax in axes:ax.grid(alpha=.25)
    fig.suptitle('Encoder + illustrative IMU measurements (analytic motion, no slip)')
    fig.tight_layout();fig.savefig(args.output/'sensor_diagnostics.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
