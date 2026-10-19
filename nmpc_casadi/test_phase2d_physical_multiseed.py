"""Self-contained synthetic regression tests for Phase2D multiseed aggregator.

Run from nmpc_casadi: python -m unittest -v test_phase2d_physical_multiseed
No NMPC solver is required to run these tests.
"""
import csv
import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import phase2d_physical_multiseed as study


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mock_seed(root, seed, *, all_scens=None, all_offsets=None):
    directory = root / f'seed_{seed}'
    directory.mkdir()
    scenarios = study.SCENARIOS if all_scens is None else all_scens
    offsets = study.OFFSETS if all_offsets is None else all_offsets
    cfg = {
        'scenario_set':'physical','sampling_seed':seed,'seed':0,
        'qfpsi_signed_sweep':True,'split_critic':True,
        'ablation_components':True,'rollout_steps':16,
        'seconds':150.0,'actor_lr':.001,'critic_lr':.001,
        'checkpoint_source': str(root / 'original.npz'),
        'evaluation_path_is_held_out':True,
        'evaluation_reference':{'id':'cad_eval_external',
                                'user_supplied_evaluation_track':str(root / 'cad.csv')},
        'controllers':list(study.OFFSETS.values()),
    }
    (directory / 'config.json').write_text(json.dumps(cfg),encoding='utf-8')
    parity, signed, comparison = [], [], []
    for scenario in scenarios:
        parity.append({
            'scenario':scenario,'n_steps':1434,
            'max_abs_weight_diff':'0.0','max_abs_cte_diff_m':'0.0',
            'matched_full_weights':'True','matched_full_cte':'True',
        })
        base_mean = 1.3 + .01*seed + .02*list(study.SCENARIOS).index(scenario)
        base_max = 3.3 + .02*seed
        base_heading = .06 + .0001*seed
        base_reward = -900. + seed
        for offset, controller in study.OFFSETS.items():
            if offset not in offsets:
                continue
            mean_cte = base_mean + offset*.011
            max_cte = base_max + offset*.021
            heading = base_heading - offset*.0001
            reward = base_reward + offset*.4
            signed.append({
                'scenario':scenario,'controller':controller,
                'signed_offset_pct':offset,
                'mean_cte_post_cm':mean_cte,
                'max_cte_post_cm':max_cte,
                'heading_rmse_post_rad':heading,'reward':reward,
                'reached_end':'True',
                'solver_success_pct':100,'updates':90,
                'delta_mean_cte_post_cm_vs_zero':mean_cte-base_mean,
                'delta_max_cte_post_cm_vs_zero':max_cte-base_max,
                'delta_heading_rmse_post_rad_vs_zero':heading-base_heading,
                'delta_reward_vs_zero':reward-base_reward,
            })
            comparison.append({
                'scenario':scenario,'controller':controller,
                'mean_cte_post_m':mean_cte/100,
                'max_cte_post_m':max_cte/100,
                'rmse_heading_post_rad':heading,
                'return':reward,'reached_end':'True',
                'solver_success_pct':100,
                'deadline_miss_total_pct':.0697 if offset == -5 else 0.,
                'control_step_total_p99_ms':5.3,
                'control_step_total_max_ms':85.0 if offset == -5 else 8.,
                'updates':90,'split_qf_head_unchanged':'True',
            })
    write_csv(directory / 'signed_qfpsi_zero_parity.csv', parity)
    write_csv(directory / 'signed_qfpsi_sensitivity.csv', signed)
    write_csv(directory / 'comparison.csv', comparison)
    return directory


class Phase2DValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'original.npz').write_bytes(b'example checkpoint fixture')
        (self.root / 'cad.csv').write_text('s,X,Y\n0,0,0\n',encoding='utf-8')

    def tearDown(self):
        self.tmp.cleanup()

    def call_seed(self, seed_dir, seed):
        return study.inspect_seed(
            seed_dir,seed,checkpoint=self.root / 'original.npz',
            track_csv=self.root / 'cad.csv',
        )

    def test_five_seeds_all_scenarios_and_paired_macro(self):
        rows = []
        for seed in study.DEFAULT_SEEDS:
            rows.extend(self.call_seed(mock_seed(self.root,seed),seed))
        self.assertEqual(len(rows),100)
        detail, macro = study.summarize_all(rows,study.DEFAULT_SEEDS)
        self.assertEqual(len(detail),20)
        self.assertEqual(len(macro),5)
        neg5 = next(r for r in macro if r['offset_pct']==-5)
        self.assertAlmostEqual(neg5['average_delta_mean_cte_cm'],-.055)
        self.assertEqual(neg5['mean_cte_improved_seed_count'],5)
        self.assertEqual(neg5['scenario_seed_mean_cte_improvements'],20)
        self.assertEqual(neg5['scenario_seed_pair_count'],20)
        self.assertAlmostEqual(neg5['average_delta_max_cte_cm'],-.105)
        self.assertAlmostEqual(neg5['average_delta_heading_rmse_rad'],.0005)
        self.assertEqual(neg5['all_completed'],True)
        self.assertAlmostEqual(neg5['worst_deadline_miss_pct'],.0697)
        self.assertAlmostEqual(neg5['max_control_step_ms'],85.0)
        self.assertIn('ci95_delta_mean_cte_cm_low',neg5)
        self.assertLess(neg5['ci95_delta_mean_cte_cm_high'],0)

    def test_parity_violation_rejected(self):
        d = mock_seed(self.root,11)
        path = d / 'signed_qfpsi_zero_parity.csv'
        data = study.csv_rows(path)
        data[0]['matched_full_cte']='False'
        path.unlink()
        write_csv(path,data)
        with self.assertRaisesRegex(ValueError,'zero offset fails'):
            self.call_seed(d,11)

    def test_missing_scenario_or_offset_rejected(self):
        d = mock_seed(self.root,11,all_offsets={-5,-2,0,2})
        with self.assertRaisesRegex(ValueError,'expected 20'):
            self.call_seed(d,11)

    def test_seed_mismatch_rejected(self):
        d = mock_seed(self.root,11)
        with self.assertRaisesRegex(ValueError,'sampling_seed'):
            self.call_seed(d,22)

    def test_duplicate_signed_row_rejected(self):
        d = mock_seed(self.root,11)
        path=d/'signed_qfpsi_sensitivity.csv'
        data=study.csv_rows(path)
        data[-1]=data[-2].copy()
        path.unlink()
        write_csv(path,data)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            self.call_seed(d,11)

    def test_nonfinite_metric_rejected(self):
        d=mock_seed(self.root,11)
        path=d/'signed_qfpsi_sensitivity.csv'
        data=study.csv_rows(path)
        data[0]['mean_cte_post_cm']='nan'
        path.unlink()
        write_csv(path,data)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            self.call_seed(d,11)

    def test_garbled_paired_delta_rejected(self):
        d=mock_seed(self.root,11)
        path=d/'signed_qfpsi_sensitivity.csv'
        data=study.csv_rows(path)
        data[0]['delta_max_cte_post_cm_vs_zero']='19'
        path.unlink()
        write_csv(path,data)
        with self.assertRaisesRegex(ValueError,'delta_max_cte_post_cm_vs_zero'):
            self.call_seed(d,11)

    def test_solver_failure_and_deadline_misses_are_preserved(self):
        d=mock_seed(self.root,11)
        path=d/'comparison.csv'
        data=study.csv_rows(path)
        data[0]['solver_success_pct']='99.5'
        path.unlink()
        write_csv(path,data)
        with self.assertRaisesRegex(ValueError,'solver_success_pct'):
            self.call_seed(d,11)
        path2=d/'signed_qfpsi_sensitivity.csv'
        signed=study.csv_rows(path2)
        signed[0]['solver_success_pct']='99.5'
        path2.unlink()
        write_csv(path2,signed)
        parsed=self.call_seed(d,11)
        self.assertEqual(parsed[0]['solver_success_pct'],99.5)
        self.assertEqual(parsed[0]['control_step_total_max_ms'],85.0)
        self.assertEqual(parsed[0]['deadline_miss_total_pct'],.0697)

    def test_incorrect_heldout_cad_path_rejected(self):
        d=mock_seed(self.root,11)
        cpath=d/'config.json'
        c=json.loads(cpath.read_text())
        c['evaluation_reference']['user_supplied_evaluation_track']='different.csv'
        cpath.write_text(json.dumps(c))
        with self.assertRaisesRegex(ValueError,'CAD source path mismatch'):
            self.call_seed(d,11)

    def test_config_validation_and_unsupported_output_rejected(self):
        d=mock_seed(self.root,11)
        path=d/'config.json'
        c=json.loads(path.read_text())
        c['split_critic']=False
        path.write_text(json.dumps(c))
        with self.assertRaisesRegex(ValueError,'split_critic'):
            self.call_seed(d,11)

    def test_summary_only_never_reruns_or_overwrites_seed_files(self):
        study_root = self.root / 'study_results'
        study_root.mkdir()
        (study_root / 'original.npz').write_bytes(b'example checkpoint fixture')
        (study_root / 'cad.csv').write_text('s,X,Y\n0,0,0\n', encoding='utf-8')
        for seed in study.DEFAULT_SEEDS:
            mock_seed(study_root, seed)
        fake_script = self.root / 'phase2d_physical_multiseed.py'
        fake_script.write_text('script stub')
        (self.root / 'adapt_nmpc.py').write_text('evaluator stub')
        argv = [
            '--checkpoint', str(study_root/'original.npz'),
            '--track-csv', str(study_root/'cad.csv'),
            '--output-root', str(study_root),
            '--summary-only',
        ]
        with patch.object(study, '__file__', str(fake_script)):
            first = study.execute(study.parse_args(argv))
            second = study.execute(study.parse_args(argv))
        self.assertNotEqual(first,second)
        self.assertTrue((first/'phase2d_macro_by_offset.csv').is_file())
        self.assertTrue((first/'phase2d_scenario_offset_summary.csv').is_file())
        self.assertTrue((first/'phase2d_seed_scenario_offset.csv').is_file())
        self.assertEqual(len(study.csv_rows(first/'phase2d_seed_scenario_offset.csv')), 100)
        self.assertEqual(len(study.csv_rows(first/'phase2d_scenario_offset_summary.csv')), 20)
        self.assertEqual(len(study.csv_rows(first/'phase2d_macro_by_offset.csv')), 5)
        self.assertTrue((second/'phase2d_provenance.json').is_file())

    def test_dry_run_does_not_write_results(self):
        fake_script = self.root / 'phase2d_physical_multiseed.py'
        fake_script.write_text('stub')
        (self.root / 'adapt_nmpc.py').write_text('stub')
        output = self.root / 'new_eval_results'
        argv = [
            '--checkpoint', str(self.root/'original.npz'),
            '--track-csv', str(self.root/'cad.csv'),
            '--output-root', str(output),
            '--dry-run',
        ]
        with patch.object(study, '__file__', str(fake_script)):
            self.assertIsNone(study.execute(study.parse_args(argv)))
        self.assertFalse(output.exists())

    def test_summary_dir_never_overwrites(self):
        a=study.unique_summary_dir(self.root)
        (a/'sentinel.txt').write_text('never overwrite')
        b=study.unique_summary_dir(self.root)
        self.assertNotEqual(a,b)
        self.assertEqual((a/'sentinel.txt').read_text(),'never overwrite')

    def test_no_false_ci_for_only_one_seed(self):
        d=mock_seed(self.root,11)
        rows=self.call_seed(d,11)
        detail, macro=study.summarize_all(rows,[11])
        self.assertNotIn('ci95_delta_mean_cte_cm_low',macro[0])


if __name__=='__main__':
    unittest.main()
