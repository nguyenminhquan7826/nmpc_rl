import importlib.util,math,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('turn',Path(__file__).parents[1]/'tools/audit_turns.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TurnAuditTests(unittest.TestCase):
 def test_known_90_degree_turn_with_bias(self):
  t=[k*.02 for k in range(101)];y=[math.radians(45*x) for x in t];r=[math.radians(45.4)]*len(t)
  a=m.measure(t,y,r,{'name':'left','start_s':0,'end_s':2,'reference_deg':90},.4)
  self.assertAlmostEqual(a['heading_delta_deg'],90)
  self.assertAlmostEqual(a['gyro_integral_deg'],90.8)
  self.assertAlmostEqual(a['gyro_bias_corrected_integral_deg'],90)
  self.assertAlmostEqual(a['gyro_corrected_error_vs_reference_deg'],0)
 def test_interpolated_boundaries_and_negative_turn(self):
  a=m.measure([0,1,2],[0,-math.pi/4,-math.pi/2],[-math.pi/4]*3,{'start_s':.5,'end_s':1.5,'reference_deg':-45},0)
  self.assertAlmostEqual(a['heading_delta_deg'],-45)
  self.assertAlmostEqual(a['gyro_integral_deg'],-45)
 def test_no_ground_truth_is_not_invented(self):
  a=m.measure([0,1],[0,.2],[.2,.2],{'start_s':0,'end_s':1},0)
  self.assertIsNone(a['gyro_error_vs_reference_deg'])
 def test_clock_ratio_corrects_time_not_heading_or_rate(self):
  ratio=1.02;t=[k*.02*ratio for k in range(101)]
  y=[math.radians(90*k/100) for k in range(101)];r=[math.radians(45.4)]*len(t)
  a=m.measure(t,y,r,{'start_s':0,'end_s':t[-1],'reference_deg':90},.4,source_clock_ratio=ratio)
  self.assertAlmostEqual(a['heading_delta_deg'],90)
  self.assertAlmostEqual(a['gyro_integral_deg'],90.8*ratio)
  self.assertAlmostEqual(a['gyro_time_corrected_integral_deg'],90.8)
  self.assertAlmostEqual(a['gyro_bias_corrected_integral_deg'],90)
  self.assertAlmostEqual(a['time_corrected_duration_s'],2)
 def test_invalid_clock_ratios_rejected(self):
  for ratio in [0,-1,float('nan'),float('inf')]:
   with self.assertRaises(ValueError):m.measure([0,1],[0,0],[0,0],{'start_s':0,'end_s':1},0,source_clock_ratio=ratio)
 def test_invalid_window_rejected(self):
  with self.assertRaises(ValueError):m.measure([0,1],[0,0],[0,0],{'start_s':0,'end_s':2},0)
if __name__=='__main__':unittest.main()
