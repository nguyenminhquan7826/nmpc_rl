"""Planar encoder/fused-yaw odometry; assumes no lateral slip.

Initial global pose must be supplied/calibrated externally. No true plant
state is used after initialization. Gyro is not fused a second time with
BNO055 orientation in this minimal estimator.
"""
import numpy as np
from sensor_models import wrap_angle


class EncoderImuOdometry:
    def __init__(self, metres_per_count, initial_pose=(0.,0.,0.)):
        if not np.isfinite(metres_per_count) or metres_per_count<=0:
            raise ValueError('metres_per_count must be positive finite')
        self.metres_per_count=float(metres_per_count)
        self.reset(initial_pose)

    def reset(self, initial_pose=(0.,0.,0.)):
        pose=np.asarray(initial_pose,float)
        if pose.shape!=(3,) or not np.isfinite(pose).all():raise ValueError('Invalid initial pose')
        self.x,self.y=map(float,pose[:2]);self.yaw=float(wrap_angle(pose[2]))

    def update(self, encoder_delta, yaw_measured_rad, *, interval_distance_m=None):
        """Signed count delta plus endpoint fused yaw; circular midpoint heading.

        Optional interval_distance_m lets callers assess filtered-speed
        integration separately. It must be derived from measurements.
        """
        if not np.isfinite([encoder_delta,yaw_measured_rad]).all():raise ValueError('Nonfinite measurement')
        if int(encoder_delta)!=encoder_delta:raise ValueError('Encoder delta must be integer')
        ds=float(encoder_delta)*self.metres_per_count if interval_distance_m is None else float(interval_distance_m)
        if not np.isfinite(ds):raise ValueError('Nonfinite interval distance')
        turn=wrap_angle(yaw_measured_rad-self.yaw)
        midpoint=self.yaw+.5*turn
        self.x+=ds*np.cos(midpoint);self.y+=ds*np.sin(midpoint)
        self.yaw=float(wrap_angle(yaw_measured_rad))
        return np.array([self.x,self.y,self.yaw])
