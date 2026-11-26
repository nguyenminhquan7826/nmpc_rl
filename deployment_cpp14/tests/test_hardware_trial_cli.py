"""CLI-only tests: no serial access or motor commands."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "hardware_trial", Path(__file__).resolve().parents[1] / "tools/hardware_trial.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Tests(unittest.TestCase):
    def test_defaults(self):
        p = module.parser()
        a = p.parse_args(["prepare"])
        self.assertEqual(a.start, "initial_heads")
        self.assertEqual(a.horizon, 20)
        self.assertEqual(a.hardware.name, "hardware_nominal.json")
        self.assertEqual(a.reference.name, "reference_straight_2m.csv")

    def test_one_mode_and_drive_interlock(self):
        p = module.parser()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            binary = root / "hardware_runner"
            binary.touch()
            binary.chmod(0o755)
            for name in ("initial.model", "hardware.settings", "reference.csv"):
                (root / name).touch()
            base = ["run", "critic_adapt", "--inputs", d, "--binary", str(binary),
                    "--output", str(root / "new")]
            cmd, output = module.run_command(p.parse_args(base), p)
            self.assertEqual(cmd[cmd.index("--mode") + 1], "critic_adapt")
            self.assertEqual(cmd.count("--mode"), 1)
            self.assertNotIn("--drive", cmd)
            self.assertEqual(cmd[cmd.index("--device") + 1], "/dev/ttyACM0")
            self.assertFalse(output.exists())
            with self.assertRaises(SystemExit):
                module.run_command(p.parse_args(base + ["--drive"]), p)
            cmd, _ = module.run_command(
                p.parse_args(base + ["--drive", "--pose", "0", "0", "0"]), p)
            self.assertIn("--keyboard", cmd)
            self.assertIn("--drive", cmd)
            self.assertEqual(cmd[cmd.index("--yaw0-deg") + 1], "0.0")
            with self.assertRaises(SystemExit):
                module.run_command(p.parse_args(base + ["--drive", "--pose",
                                                        "0", "0", "0", "--input", "raw.log"]), p)


if __name__ == "__main__":
    unittest.main()
