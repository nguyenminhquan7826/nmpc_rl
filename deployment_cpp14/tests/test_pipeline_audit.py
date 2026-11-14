import importlib.util,sys,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('pipeline',Path(__file__).parents[1]/'tools/audit_sensor_pipeline.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class AuditTests(unittest.TestCase):
 def test_missing_diagnostic_columns_rejected(self):
  with self.assertRaises(ValueError):m.audit([],[{'accepted':'1'}],1)
 def test_frame_count_mismatch_detected(self):
  a=m.audit([['unused']],[],1);self.assertFalse(a['mapping_and_pose_passed'])
if __name__=='__main__':unittest.main()
