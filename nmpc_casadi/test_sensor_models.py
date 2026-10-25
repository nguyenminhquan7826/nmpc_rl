import unittest
import numpy as np
from sensor_models import SensorConfig,VehicleSensors

class SensorTests(unittest.TestCase):
    def test_conversion_and_fractional_counts(self):
        c=SensorConfig();self.assertAlmostEqual(c.counts_per_wheel_rev,836*38/13)
        self.assertAlmostEqual(c.metres_per_count/c.sample_period_s,.00836,places=5)
        s=VehicleSensors(c);r=[s.sample(c.metres_per_count*.3,0,0) for _ in range(101)]
        self.assertEqual(r[-1]['encoder_count'],30)
        self.assertEqual(sum(v['encoder_delta'] for v in r),30)
        self.assertLess(abs(s.distance-r[-1]['encoder_distance_m']),c.metres_per_count)
        self.assertEqual(s.sample(-c.metres_per_count*10,0,0)['encoder_delta'],-10)

    def test_reproducibility(self):
        a,b,c=VehicleSensors(seed=2),VehicleSensors(seed=2),VehicleSensors(seed=3)
        ar=[a.sample(.001,.3,.02) for _ in range(10)]
        br=[b.sample(.001,.3,.02) for _ in range(10)]
        cr=[c.sample(.001,.3,.02) for _ in range(10)]
        self.assertEqual(ar,br);self.assertNotEqual(ar[0]['yaw_measured_rad'],cr[0]['yaw_measured_rad'])
        self.assertEqual([r['encoder_count'] for r in ar],[r['encoder_count'] for r in cr])
        a.reset();self.assertEqual(a.sample(.001,.3,.02),ar[0])

    def test_clean_imu_and_validation(self):
        c=SensorConfig(yaw_noise_std_rad=0,yaw_rate_noise_std_rad_s=0,yaw_bias_rad=0,
            yaw_rate_bias_rad_s=0,yaw_bias_walk_std_rad_sqrt_s=0,rate_bias_walk_std_rad_s_sqrt_s=0,
            velocity_filter_tau_s=0)
        r=VehicleSensors(c).sample(.001,2*np.pi+.2,.03)
        self.assertAlmostEqual(r['yaw_measured_rad'],.2)
        self.assertEqual(r['yaw_rate_measured_rad_s'],.03)
        self.assertEqual(r['vx_filtered_mps'],r['vx_raw_mps'])
        with self.assertRaises(ValueError):VehicleSensors().sample(np.nan,0,0)
        with self.assertRaises(ValueError):SensorConfig(velocity_filter_tau_s=-1)

if __name__=='__main__':unittest.main()
