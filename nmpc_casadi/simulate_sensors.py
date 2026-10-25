"""Sensor-only diagnostic using analytic motion; not a controller evaluation."""
import argparse,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from sensor_models import SensorConfig,VehicleSensors,wrap_angle
from encoder_imu_odometry import EncoderImuOdometry


def simulate(config,seed,duration):
    sensor=VehicleSensors(config,seed);rows=[];dt=config.sample_period_s
    odom=EncoderImuOdometry(config.metres_per_count)
    filtered=EncoderImuOdometry(config.metres_per_count)
    true_xy=np.zeros(2)
    # Analytic travel: smooth start to ~0.1 m/s and a slow speed variation.
    def distance(t):return .1*(t-.3*(1-np.exp(-t/.3)))+.01/.4*(1-np.cos(.4*t))
    for k in range(int(duration/dt)):
        t=(k+1)*dt;s=distance(t);delta=s-distance(t-dt)
        yaw=.3*np.sin(.25*t);rate=.075*np.cos(.25*t)
        row=sensor.sample(delta,yaw,rate)
        # Ground truth uses the same no-slip kinematics, with true midpoint yaw.
        midpoint=.3*np.sin(.25*(t-.5*dt))
        true_xy+=delta*np.array([np.cos(midpoint),np.sin(midpoint)])
        measured=odom.update(row['encoder_delta'],row['yaw_measured_rad'])
        smoothed=filtered.update(row['encoder_delta'],row['yaw_measured_rad'],
            interval_distance_m=row['vx_filtered_mps']*dt)
        row.update(X_true_m=true_xy[0],Y_true_m=true_xy[1],
                   X_odom_m=measured[0],Y_odom_m=measured[1],
                   X_odom_filtered_m=smoothed[0],Y_odom_filtered_m=smoothed[1],
                   position_error_m=float(np.linalg.norm(measured[:2]-true_xy)),
                   position_error_filtered_m=float(np.linalg.norm(smoothed[:2]-true_xy)))
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
    summary=dict(role='analytic sensor and odometry diagnostic, no controller evaluation',
        odometry_assumptions='known initial pose, no lateral slip; raw encoder + fused yaw midpoint; filtered-speed alternative reported separately',
        odometry_final_error_m=float(d.position_error_m.iloc[-1]),
        odometry_rmse_m=float(np.sqrt(np.mean(d.position_error_m**2))),
        odometry_max_error_m=float(d.position_error_m.max()),
        filtered_odometry_final_error_m=float(d.position_error_filtered_m.iloc[-1]),seed=args.seed,
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
    fig,axes=plt.subplots(3,1,figsize=(11,12))
    for x,y,label in [('X_true_m','Y_true_m','True'),('X_odom_m','Y_odom_m','Raw counts + measured yaw'),
                      ('X_odom_filtered_m','Y_odom_filtered_m','Filtered speed + measured yaw')]:
        axes[0].plot(d[x],d[y],label=label)
    axes[0].axis('equal');axes[0].set(xlabel='X (m)',ylabel='Y (m)');axes[0].legend()
    axes[1].plot(d.t_s,1000*(d.X_odom_m-d.X_true_m),label='X error')
    axes[1].plot(d.t_s,1000*(d.Y_odom_m-d.Y_true_m),label='Y error')
    axes[1].set(xlabel='Time (s)',ylabel='Signed error (mm)');axes[1].legend()
    axes[2].plot(d.t_s,1000*d.position_error_m,label='Raw-count position error')
    axes[2].plot(d.t_s,1000*d.position_error_filtered_m,label='Filtered-speed position error')
    axes[2].set(xlabel='Time (s)',ylabel='2D error norm (mm)');axes[2].legend()
    for ax in axes:ax.grid(alpha=.25)
    fig.suptitle('Encoder/IMU odometry: known initial pose, no lateral slip')
    fig.tight_layout();fig.savefig(args.output/'odometry_diagnostics.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
