import importlib.util,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
class Tests(unittest.TestCase):
 def test_settings_are_hardware_specific_and_reject_fast_route(self):
  spec=importlib.util.spec_from_file_location('prepare_hardware',ROOT/'deployment_cpp14/tools/prepare_hardware_trial.py');self.assertIsNotNone(spec)
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  import json
  c=json.loads((ROOT/'deployment_cpp14/examples/hardware_nominal.json').read_text())
  with tempfile.TemporaryDirectory() as d:
   route=Path(d)/'route.csv';route.write_text('s_m,vx_ref_mps,kappa_1_per_m\n0,0.1,0\n1,0.1,0\n')
   x=module.settings(c,route,{'Qe_fixed':[1]*6,'Re_fixed':[1]*2});self.assertEqual(x['state_lower'][0],0);self.assertAlmostEqual(sum(x['model'][4:6]),.25);self.assertLess(x['input_upper'][1],.4)
   route.write_text('s_m,vx_ref_mps,kappa_1_per_m\n0,0.5,0\n1,0.5,0\n')
   with self.assertRaises(ValueError):module.settings(c,route,{'Qe_fixed':[1]*6,'Re_fixed':[1]*2})
   route.write_text('s_m,vx_ref_mps,kappa_1_per_m\n0,0.1,0\n1,0.1,-2\n')
   with self.assertRaisesRegex(ValueError,'CAD steering exceeds limit at s=1 m'):
    module.settings(c,route,{'Qe_fixed':[1]*6,'Re_fixed':[1]*2})
if __name__=='__main__':unittest.main()
