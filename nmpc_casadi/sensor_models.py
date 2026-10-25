"""Illustrative encoder + BNO055 measurement simulator, independent of plant.

IMU parameters are experiment assumptions, not BNO055 datasheet/calibration
claims. No true state is modified. A caller supplies interval travel and the
endpoint orientation/rate every encoder sample (default 10 ms).
"""
from dataclasses import dataclass,asdict
import numpy as np


def wrap_angle(angle):
    return (angle+np.pi)%(2*np.pi)-np.pi


@dataclass(frozen=True)
class SensorConfig:
    sample_period_s: float = .01
    wheel_radius_m: float = .0325
    encoder_cycles_per_motor_rev: int = 11
    quadrature_factor: int = 4
    gearbox_ratio: float = 19.
    differential_ratio: float = 38./13.
    velocity_filter_tau_s: float = .05
    yaw_noise_std_rad: float = np.deg2rad(.5)
    yaw_rate_noise_std_rad_s: float = .01
    yaw_bias_rad: float = np.deg2rad(.2)
    yaw_rate_bias_rad_s: float = .002
    yaw_bias_walk_std_rad_sqrt_s: float = np.deg2rad(.02)
    rate_bias_walk_std_rad_s_sqrt_s: float = .0002

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Sensor settings must be finite')
        for name in ('sample_period_s','wheel_radius_m','encoder_cycles_per_motor_rev',
                     'quadrature_factor','gearbox_ratio','differential_ratio'):
            if getattr(self,name)<=0:raise ValueError(name+' must be positive')
        for name in ('encoder_cycles_per_motor_rev','quadrature_factor'):
            if int(getattr(self,name))!=getattr(self,name):raise ValueError(name+' must be an integer')
        for name in ('velocity_filter_tau_s','yaw_noise_std_rad','yaw_rate_noise_std_rad_s',
                     'yaw_bias_walk_std_rad_sqrt_s','rate_bias_walk_std_rad_s_sqrt_s'):
            if getattr(self,name)<0:raise ValueError(name+' must be nonnegative')

    @property
    def counts_per_wheel_rev(self):
        return self.encoder_cycles_per_motor_rev*self.quadrature_factor*self.gearbox_ratio*self.differential_ratio

    @property
    def metres_per_count(self):
        return 2*np.pi*self.wheel_radius_m/self.counts_per_wheel_rev


class VehicleSensors:
    def __init__(self, config=None, seed=0):
        self.config=config or SensorConfig()
        self.seed=int(seed)
        self.reset()

    def reset(self):
        self.rng=np.random.default_rng(self.seed)
        self.distance=0.;self.previous_count=0;self.filtered_speed=0.;self.time=0.
        self.yaw_bias=self.config.yaw_bias_rad
        self.rate_bias=self.config.yaw_rate_bias_rad_s

    def sample(self, interval_distance_m, yaw_true_rad, yaw_rate_true_rad_s):
        """One fixed-period measurement; preserve sub-count travel and direction.

        Encoder phase is zero at reset. Speed filter starts at zero. interval
        distance is signed longitudinal wheel travel (no slip in this model).
        Yaw is a simplified fused-orientation measurement, not raw accel/mag.
        """
        if not np.isfinite([interval_distance_m,yaw_true_rad,yaw_rate_true_rad_s]).all():
            raise ValueError('Nonfinite sensor input')
        c=self.config;dt=c.sample_period_s
        self.distance+=float(interval_distance_m)
        count=int(np.floor(self.distance/c.metres_per_count))
        change=count-self.previous_count;self.previous_count=count
        raw_speed=change*c.metres_per_count/dt
        alpha=1. if c.velocity_filter_tau_s==0 else -np.expm1(-dt/c.velocity_filter_tau_s)
        self.filtered_speed+=alpha*(raw_speed-self.filtered_speed)
        self.yaw_bias+=c.yaw_bias_walk_std_rad_sqrt_s*np.sqrt(dt)*self.rng.standard_normal()
        self.rate_bias+=c.rate_bias_walk_std_rad_s_sqrt_s*np.sqrt(dt)*self.rng.standard_normal()
        yaw=wrap_angle(yaw_true_rad+self.yaw_bias+c.yaw_noise_std_rad*self.rng.standard_normal())
        rate=yaw_rate_true_rad_s+self.rate_bias+c.yaw_rate_noise_std_rad_s*self.rng.standard_normal()
        self.time+=dt
        return dict(t_s=self.time,encoder_count=count,encoder_delta=change,
                    vx_raw_mps=raw_speed,vx_filtered_mps=self.filtered_speed,
                    encoder_distance_m=count*c.metres_per_count,
                    yaw_measured_rad=float(yaw),yaw_rate_measured_rad_s=float(rate),
                    yaw_bias_rad=self.yaw_bias,yaw_rate_bias_rad_s=self.rate_bias)
