import unittest
import numpy as np
from encoder_imu_odometry import EncoderImuOdometry
from sensor_models import SensorConfig
from simulate_sensors import simulate

class OdometryTests(unittest.TestCase):
    def test_straight_reverse_and_reset(self):
        o=EncoderImuOdometry(.001)
        np.testing.assert_allclose(o.update(10,0),[.01,0,0])
        np.testing.assert_allclose(o.update(-5,0),[.005,0,0])
        o.reset((1,2,np.pi/2))
        np.testing.assert_allclose(o.update(10,np.pi/2),[1,2.01,np.pi/2])

    def test_angle_wrap_midpoint(self):
        o=EncoderImuOdometry(.01,(0,0,np.deg2rad(179)))
        pose=o.update(1,np.deg2rad(-179))
        np.testing.assert_allclose(pose[:2],[-.01,0],atol=1e-12)
        with self.assertRaises(ValueError):o.update(.5,0)

    def test_noise_free_kinematic_accuracy(self):
        c=SensorConfig(yaw_noise_std_rad=0,yaw_rate_noise_std_rad_s=0,yaw_bias_rad=0,
            yaw_rate_bias_rad_s=0,yaw_bias_walk_std_rad_sqrt_s=0,rate_bias_walk_std_rad_s_sqrt_s=0)
        d=simulate(c,0,30)
        self.assertLess(d.position_error_m.max(),2*c.metres_per_count)
        np.testing.assert_array_equal(simulate(c,4,1).X_odom_m,simulate(c,2,1).X_odom_m)

if __name__=='__main__':unittest.main()
