import importlib.util,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('clock_audit',Path(__file__).parents[1]/'tools/audit_clock_rate.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def fixture(ratio=1.02,batch=1,wrap=False):
 rows=[];base=2**60;start=2**32-1000 if wrap else 500000
 for k in range(6001):
  group=k//batch;host=group*batch*.02
  ns=base+round(host*1e9)
  rows.append({'accepted':'1','tx_ms':str((start+round(k*.02*ratio*1000))%2**32),'host_rx_monotonic_ns':str(ns),'host_rx_batch_id':str(group+1)})
 return rows
class ClockAuditTests(unittest.TestCase):
 def test_known_clock_rate_large_integer_host_epoch(self):
  a=m.analyze(fixture());self.assertAlmostEqual(a['apparent_stm32_seconds_per_host_second'],1.02,places=5)
  self.assertAlmostEqual(a['apparent_drift_ppm'],20000,delta=5)
 def test_batched_rows_do_not_create_fake_receive_times(self):
  a=m.analyze(fixture(batch=10));self.assertAlmostEqual(a['apparent_stm32_seconds_per_host_second'],1.02,places=4)
  self.assertEqual(a['max_frames_per_batch'],10)
 def test_mcu_counter_rollover(self):
  self.assertAlmostEqual(m.analyze(fixture(wrap=True))['apparent_stm32_seconds_per_host_second'],1.02,places=5)
 def test_refuse_missing_or_short_or_reset_time(self):
  for rows in ([{'accepted':'1'}],fixture()[:100]):
   with self.assertRaises(ValueError):m.analyze(rows)
  rows=fixture();rows[100]['tx_ms']='1'
  with self.assertRaises(ValueError):m.analyze(rows)
 def test_same_batch_must_have_identical_timestamp(self):
  rows=fixture(batch=10);rows[2]['host_rx_monotonic_ns']=str(int(rows[2]['host_rx_monotonic_ns'])+1)
  with self.assertRaises(ValueError):m.analyze(rows)
 def test_epoch_does_not_change_slope(self):
  rows=fixture();a=m.analyze(rows)
  for r in rows:r['host_rx_monotonic_ns']=str(int(r['host_rx_monotonic_ns'])+1234567890123456789)
  self.assertEqual(a['apparent_stm32_seconds_per_host_second'],m.analyze(rows)['apparent_stm32_seconds_per_host_second'])
if __name__=='__main__':unittest.main()
