"""Offline analysis regression using explicit synthetic logs, no serial."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1]/"tools"/"analyze_hardware.py"

class Tests(unittest.TestCase):
    def test_metrics_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            run = p/"drive_fixed_100_1"
            run.mkdir()
            (run/"summary.txt").write_text("mode=fixed\ndrive=1\nseed=0\nreason=duration_completed\ncontrol_steps=2\nfeedback_count=2\nactor_updates=0\ncritic_updates=0\nfinal_stop_confirmed=1\n")
            (run/"control.csv").write_text(
                "host_s,step,mode,X,Y,yaw,vx,progress,cte,solver_success,clipped,solve_ms,reference0,reference2,reference4,reference5,control_compute_ms\n"
                "2,1,fixed,0,0.03,6.283185307179586,0.08,0,0.03,1,0,5,0.1,0,0,0,6\n"
                "2.08,2,fixed,0.01,0.04,0,0.1,0.01,0.04,1,0,5,0.1,0,0.01,0,7\n")
            (run/"reference.csv").write_text("s_m,X_ref_m,Y_ref_m\n0,0,0\n2,2,0\n")
            (run/"transitions.csv").write_text("host_s,reward,td,batch_size\n2.08,-1,0,0\n2.16,-3,0,0\n")
            (run/"events.csv").write_text("host_s,event,sequence,detail\n2,tx_RUN,1,requested\n2.08,tx_RUN,2,requested\n2.16,tx_STOP,3,requested\n")
            for k in ("initial.model","settings.txt"):
                (run/k).write_text("fixture")
            out = p/"analysis"
            args = [sys.executable,str(TOOL),str(run),"--condition","bench","--output",str(out),"--no-plots"]
            proc = subprocess.run(args,capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            with (out/"summary.csv").open() as f: m=next(csv.DictReader(f))
            self.assertAlmostEqual(float(m["cte_m_rmse"]),(.03**2/2+.04**2/2)**.5)
            self.assertAlmostEqual(float(m["heading_abs_error_deg_rmse"]),0)
            self.assertAlmostEqual(float(m["active_duration_s"]),.16)
            self.assertEqual(float(m["total_reward"]),-4)
            self.assertEqual(float(m["control_compute_ms_max"]),7)
            checks=json.loads((out/"comparison_checks.json").read_text())
            self.assertFalse(checks["all_designated_vehicle"])
            self.assertFalse(checks["ground_truth_available"])
            self.assertNotEqual(subprocess.run(args,capture_output=True).returncode,0)

if __name__ == "__main__":
    unittest.main()
