"""Phase 2D multi-seed / physical-plant confirmation (evaluation ONLY).

Runs the UNMODIFIED adapt_nmpc.py with five signed pretrained Qf_psi
interventions, across 4 physical scenarios and 5 independent sampling
seeds. Strictly never overwrites prior seed output directories.

Requirements: existing nmpc_casadi/adapt_nmpc.py, its runtime dependencies.
This wrapper/aggregator itself uses ONLY the Python standard library.

Examples (PowerShell, from nmpc_casadi):
    python phase2d_physical_multiseed.py --checkpoint $ck --track-csv $cad
    python phase2d_physical_multiseed.py --checkpoint $ck --track-csv $cad --summary-only
    python phase2d_physical_multiseed.py --checkpoint $ck --track-csv $cad --resume

The standalone evaluation input XREF and selected pretrained checkpoint
are READ-ONLY. Do not interpret 100% solver success as hard real-time or
closed-loop stability. This study uses one CAD geometry, NOT 20 tracks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


SCENARIOS = ('nominal', 'payload_ramp', 'tire_segment', 'rolling_segment')
OFFSETS = {
    -5: 'ac_nmpc_split_qfpsi_m05',
    -2: 'ac_nmpc_split_qfpsi_m02',
    0: 'ac_nmpc_split_qfpsi_p00',
    2: 'ac_nmpc_split_qfpsi_p02',
    5: 'ac_nmpc_split_qfpsi_p05',
}
BASELINE = OFFSETS[0]
DEFAULT_SEEDS = (11, 22, 33, 44, 55)
METRIC_FIELDS = ('mean_cte_post_cm', 'max_cte_post_cm',
                 'heading_rmse_post_rad', 'reward')
TIMING_FIELDS = ('deadline_miss_total_pct', 'control_step_total_p99_ms',
                 'control_step_total_max_ms')


def numeric(value, name):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as err:
        raise ValueError(f'Invalid numeric {name}: {value!r}') from err
    if not math.isfinite(parsed):
        raise ValueError(f'Nonfinite {name}: {value!r}')
    return parsed


def boolean(value, name):
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in ('true', '1'):
        return True
    if str(value).strip().lower() in ('false', '0'):
        return False
    raise ValueError(f'Invalid boolean {name}: {value!r}')


def csv_rows(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing evaluation result: {path}')
    with path.open('r', newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f'Missing CSV header: {path}')
        out = list(reader)
    if not out:
        raise ValueError(f'Empty evaluation result: {path}')
    return out


def by_key(rows, fields, label):
    result = {}
    for row in rows:
        try:
            key = tuple(row[f] for f in fields)
        except KeyError as err:
            raise ValueError(f'{label}: missing key column {err}') from err
        if key in result:
            raise ValueError(f'{label}: duplicate row key={key}')
        result[key] = row
    return result


def near(x, y, label, atol=1e-8):
    if not math.isclose(x, y, rel_tol=1e-9, abs_tol=atol):
        raise ValueError(f'{label}: {x} != {y}')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect_seed(seed_dir, seed, *, seconds=150.0, rollout_steps=16,
                 actor_lr=.001, critic_lr=.001, checkpoint=None,
                 track_csv=None):
    """Validate all four scenarios, all signed arms and full zero parity.

    Return 20 rows. Failure/completion outcomes are OBSERVED and reported,
    not excluded from the results; malformed/missing outputs raise errors.
    """
    config_path = seed_dir / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'Missing {config_path}')
    try:
        cfg = json.loads(config_path.read_text(encoding='utf-8'))
    except (ValueError, UnicodeError) as err:
        raise ValueError(f'Corrupt config: {config_path}') from err

    must_equal = {
        'scenario_set': 'physical', 'sampling_seed': int(seed),
        'seed': 0, 'qfpsi_signed_sweep': True,
        'split_critic': True, 'ablation_components': True,
        'rollout_steps': int(rollout_steps),
    }
    for field, expected in must_equal.items():
        if cfg.get(field) != expected:
            raise ValueError(
                f'{seed_dir}: config.{field}={cfg.get(field)!r}, expected {expected!r}'
            )
    for field, expected in (
        ('seconds', seconds), ('actor_lr', actor_lr), ('critic_lr', critic_lr)
    ):
        near(numeric(cfg.get(field), f'config.{field}'), float(expected),
             f'{seed_dir}: config.{field}', atol=1e-12)
    if checkpoint is not None:
        actual = Path(str(cfg.get('checkpoint_source'))).resolve()
        if actual != Path(checkpoint).resolve():
            raise ValueError(f'{seed_dir}: checkpoint mismatch ({actual})')
    if track_csv is not None:
        if not cfg.get('evaluation_path_is_held_out'):
            raise ValueError(f'{seed_dir}: evaluation track not marked held-out')
        ref = cfg.get('evaluation_reference', {})
        if ref.get('id') != 'cad_eval_external':
            raise ValueError(f'{seed_dir}: expected independent external CAD')
        staged_origin = ref.get('user_supplied_evaluation_track')
        if not staged_origin or Path(staged_origin).resolve() != Path(track_csv).resolve():
            raise ValueError(f'{seed_dir}: evaluation CAD source path mismatch')

    parity = csv_rows(seed_dir / 'signed_qfpsi_zero_parity.csv')
    if len(parity) != len(SCENARIOS):
        raise ValueError(f'{seed_dir}: expected 4 zero-parity rows, got {len(parity)}')
    seen = set()
    for p in parity:
        scenario = p['scenario']
        if scenario not in SCENARIOS or scenario in seen:
            raise ValueError(f'{seed_dir}: invalid/duplicate zero-parity scenario {scenario}')
        seen.add(scenario)
        if not (boolean(p.get('matched_full_weights'), 'matched_full_weights')
                and boolean(p.get('matched_full_cte'), 'matched_full_cte')):
            raise ValueError(f'{seed_dir}/{scenario}: zero offset fails Phase 2A parity')
        near(numeric(p.get('max_abs_weight_diff'), 'max_abs_weight_diff'), 0.,
             f'{seed_dir}/{scenario}: zero weight diff')
        near(numeric(p.get('max_abs_cte_diff_m'), 'max_abs_cte_diff_m'), 0.,
             f'{seed_dir}/{scenario}: zero CTE diff')
        if numeric(p.get('n_steps'), 'n_steps') <= 0:
            raise ValueError(f'{seed_dir}/{scenario}: empty simulation')

    signed = csv_rows(seed_dir / 'signed_qfpsi_sensitivity.csv')
    comparison = csv_rows(seed_dir / 'comparison.csv')
    if len(signed) != len(SCENARIOS) * len(OFFSETS):
        raise ValueError(f'{seed_dir}: expected 20 signed summary rows, got {len(signed)}')
    signed_dict = by_key(signed, ('scenario', 'controller'), 'signed')
    comparison_dict = by_key(comparison, ('scenario', 'controller'), 'comparison')
    if len(set(cfg.get('controllers', []))) != len(cfg.get('controllers', [])):
        raise ValueError(f'{seed_dir}: duplicated controller in config')
    if not set(OFFSETS.values()).issubset(set(cfg.get('controllers', []))):
        raise ValueError(f'{seed_dir}: signed sweep arms missing in config')

    rows = []
    for scenario in SCENARIOS:
        try:
            zero = signed_dict[(scenario, BASELINE)]
        except KeyError as err:
            raise ValueError(f'{seed_dir}/{scenario}: missing zero arm') from err
        base_vals = {metric: numeric(zero.get(metric), metric)
                     for metric in METRIC_FIELDS}
        if base_vals['mean_cte_post_cm'] <= 0:
            raise ValueError(f'{seed_dir}/{scenario}: zero baseline Mean CTE must be positive')

        for offset, controller in OFFSETS.items():
            key = (scenario, controller)
            if key not in signed_dict or key not in comparison_dict:
                raise ValueError(f'{seed_dir}: missing scenario/controller {key}')
            source = signed_dict[key]
            comparison_row = comparison_dict[key]
            near(numeric(source.get('signed_offset_pct'), 'signed_offset_pct'),
                 offset, f'{seed_dir}/{key}: offset')
            vals = {metric: numeric(source.get(metric), metric)
                    for metric in METRIC_FIELDS}
            signed_deltas = {
                'mean_cte_post_cm': 'delta_mean_cte_post_cm_vs_zero',
                'max_cte_post_cm': 'delta_max_cte_post_cm_vs_zero',
                'heading_rmse_post_rad': 'delta_heading_rmse_post_rad_vs_zero',
                'reward': 'delta_reward_vs_zero',
            }
            for metric, delta_key in signed_deltas.items():
                near(numeric(source.get(delta_key), delta_key),
                     vals[metric]-base_vals[metric],
                     f'{seed_dir}/{key}: {delta_key}')

            converted_fields = {
                'mean_cte_post_cm': ('mean_cte_post_m', 100.0),
                'max_cte_post_cm': ('max_cte_post_m', 100.0),
                'heading_rmse_post_rad': ('rmse_heading_post_rad', 1.0),
                'reward': ('return', 1.0),
            }
            for metric, (comparison_col, scale) in converted_fields.items():
                near(vals[metric], scale * numeric(
                    comparison_row.get(comparison_col), comparison_col),
                    f'{seed_dir}/{key}: comparison.{comparison_col}')

            finished = boolean(source.get('reached_end'), 'reached_end')
            if finished != boolean(comparison_row.get('reached_end'), 'reached_end'):
                raise ValueError(f'{seed_dir}/{key}: completion mismatch')
            success_pct = numeric(source.get('solver_success_pct'),
                                  'solver_success_pct')
            near(success_pct, numeric(comparison_row.get('solver_success_pct'),
                                      'solver_success_pct'),
                 f'{seed_dir}/{key}: solver_success_pct')
            if not 0 <= success_pct <= 100:
                raise ValueError(f'{seed_dir}/{key}: invalid solver success')
            timings = {k: numeric(comparison_row.get(k), k) for k in TIMING_FIELDS}
            if not 0 <= timings['deadline_miss_total_pct'] <= 100:
                raise ValueError(f'{seed_dir}/{key}: invalid deadline miss fraction')
            if (timings['control_step_total_p99_ms'] < 0
                    or timings['control_step_total_max_ms'] <
                    timings['control_step_total_p99_ms']):
                raise ValueError(f'{seed_dir}/{key}: invalid step timings')
            updates = numeric(source.get('updates'), 'updates')
            near(updates, numeric(comparison_row.get('updates'), 'updates'),
                 f'{seed_dir}/{key}: updates')
            if updates <= 0:
                raise ValueError(f'{seed_dir}/{key}: signed sweep Actor/Value did not learn')
            if not boolean(comparison_row.get('split_qf_head_unchanged'),
                           'split_qf_head_unchanged'):
                raise ValueError(f'{seed_dir}/{key}: terminal network unexpectedly trained')

            out = {
                'seed': int(seed), 'scenario': scenario,
                'offset_pct': int(offset), 'controller': controller,
                'mean_cte_cm': vals['mean_cte_post_cm'],
                'max_cte_cm': vals['max_cte_post_cm'],
                'heading_rmse_rad': vals['heading_rmse_post_rad'],
                'reward': vals['reward'],
                'baseline_mean_cte_cm': base_vals['mean_cte_post_cm'],
                'baseline_max_cte_cm': base_vals['max_cte_post_cm'],
                'baseline_heading_rmse_rad': base_vals['heading_rmse_post_rad'],
                'baseline_reward': base_vals['reward'],
                'delta_mean_cte_cm': (vals['mean_cte_post_cm'] -
                                       base_vals['mean_cte_post_cm']),
                'delta_max_cte_cm': (vals['max_cte_post_cm'] -
                                      base_vals['max_cte_post_cm']),
                'delta_heading_rmse_rad': (vals['heading_rmse_post_rad'] -
                                           base_vals['heading_rmse_post_rad']),
                'delta_reward': vals['reward'] - base_vals['reward'],
                'mean_cte_improvement_pct': (
                    100 * (base_vals['mean_cte_post_cm'] - vals['mean_cte_post_cm'])
                    / base_vals['mean_cte_post_cm']
                ),
                'completed': finished, 'solver_success_pct': success_pct,
                'updates': int(updates), **timings,
            }
            rows.append(out)
    return rows


def mean(rows, field):
    return statistics.mean(float(row[field]) for row in rows)


def summary_stats(group):
    """Compute paired statistics without treating four plant cases as 20 seeds."""
    result = {
        'n_seeds': len(group),
        'average_mean_cte_cm': mean(group, 'mean_cte_cm'),
        'average_max_cte_cm': mean(group, 'max_cte_cm'),
        'average_heading_rmse_rad': mean(group, 'heading_rmse_rad'),
        'average_reward': mean(group, 'reward'),
        'average_delta_mean_cte_cm': mean(group, 'delta_mean_cte_cm'),
        'average_delta_max_cte_cm': mean(group, 'delta_max_cte_cm'),
        'average_delta_heading_rmse_rad': mean(group, 'delta_heading_rmse_rad'),
        'average_delta_reward': mean(group, 'delta_reward'),
        'average_improvement_mean_cte_pct': mean(group, 'mean_cte_improvement_pct'),
        'mean_cte_improved_seed_count': sum(
            r['delta_mean_cte_cm'] < 0 for r in group),
        'max_cte_improved_seed_count': sum(
            r['delta_max_cte_cm'] < 0 for r in group),
        'reward_improved_seed_count': sum(r['delta_reward'] > 0 for r in group),
        'heading_improved_seed_count': sum(
            r['delta_heading_rmse_rad'] < 0 for r in group),
        'all_completed': all(r['completed'] for r in group),
        'min_solver_success_pct': min(r['solver_success_pct'] for r in group),
        'average_solver_success_pct': mean(group, 'solver_success_pct'),
        'average_deadline_miss_pct': mean(group, 'deadline_miss_total_pct'),
        'worst_deadline_miss_pct': max(r['deadline_miss_total_pct'] for r in group),
        'max_control_step_ms': max(r['control_step_total_max_ms'] for r in group),
        'average_control_p99_ms': mean(group, 'control_step_total_p99_ms'),
    }
    if len(group) >= 2:
        for field in ('delta_mean_cte_cm', 'delta_max_cte_cm',
                      'delta_heading_rmse_rad', 'delta_reward'):
            result[f'sd_{field}'] = statistics.stdev(float(r[field]) for r in group)
    else:
        for field in ('delta_mean_cte_cm', 'delta_max_cte_cm',
                      'delta_heading_rmse_rad', 'delta_reward'):
            result[f'sd_{field}'] = ''
    # Exact Student t critical for 95% two-sided CI, df = 4 (five seeds).
    # Do NOT return a fake CI for arbitrary group sample sizes.
    if len(group) == 5:
        for field in ('delta_mean_cte_cm', 'delta_max_cte_cm'):
            uncertainty = 2.7764451051977987 * result[f'sd_{field}'] / math.sqrt(5)
            result[f'ci95_{field}_low'] = result[f'average_{field}'] - uncertainty
            result[f'ci95_{field}_high'] = result[f'average_{field}'] + uncertainty
    return result


def summarize_all(rows, seeds):
    if len(rows) != len(seeds) * len(SCENARIOS) * len(OFFSETS):
        raise ValueError('Incomplete study rows')
    by_scenario_offset = defaultdict(list)
    by_offset_seed = defaultdict(list)
    for row in rows:
        by_scenario_offset[(row['scenario'], row['offset_pct'])].append(row)
        by_offset_seed[(row['offset_pct'], row['seed'])].append(row)
    detailed = []
    for scenario in SCENARIOS:
        for offset in OFFSETS:
            group = by_scenario_offset[(scenario, offset)]
            if len(group) != len(seeds) or len({r['seed'] for r in group}) != len(seeds):
                raise ValueError(f'Missing/duplicate seed in {scenario}/{offset}')
            detailed.append({'scenario': scenario, 'offset_pct': offset,
                             **summary_stats(group)})
    macro_rows = []
    for offset in OFFSETS:
        # Four scenarios within ONE seed are strongly correlated, so first
        # aggregate within seed, THEN treat the five seed macro-results as n=5.
        seed_means = []
        for seed in seeds:
            group = by_offset_seed[(offset, seed)]
            if len(group) != len(SCENARIOS) or {
                    r['scenario'] for r in group} != set(SCENARIOS):
                raise ValueError(f'Incomplete/duplicated scenarios in {offset}/{seed}')
            composite = {'offset_pct': offset, 'seed': seed}
            floats = (
                'mean_cte_cm', 'max_cte_cm', 'heading_rmse_rad', 'reward',
                'delta_mean_cte_cm', 'delta_max_cte_cm',
                'delta_heading_rmse_rad', 'delta_reward',
                'mean_cte_improvement_pct', 'solver_success_pct',
                'deadline_miss_total_pct', 'control_step_total_p99_ms',
            )
            for field in floats:
                composite[field] = mean(group, field)
            composite['completed'] = all(r['completed'] for r in group)
            composite['control_step_total_max_ms'] = max(
                r['control_step_total_max_ms'] for r in group)
            seed_means.append(composite)
        macro_rows.append({
            'offset_pct': offset,
            'aggregation': '4_scenarios_averaged_within_each_seed_then_5_seeds',
            **summary_stats(seed_means),
            'scenario_seed_pair_count': len(seeds) * len(SCENARIOS),
            'scenario_seed_mean_cte_improvements': sum(
                r['delta_mean_cte_cm'] < 0 for r in rows
                if r['offset_pct'] == offset
            ),
            'scenario_seed_max_cte_improvements': sum(
                r['delta_max_cte_cm'] < 0 for r in rows
                if r['offset_pct'] == offset
            ),
        })
    return detailed, macro_rows


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows to save: {path}')
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def unique_summary_dir(root):
    i = 1
    while (root / f'phase2d_summary_{i:03d}').exists():
        i += 1
    target = root / f'phase2d_summary_{i:03d}'
    target.mkdir(parents=True, exist_ok=False)
    return target


def make_command(args, seed, out_dir, evaluator):
    return [
        str(args.python), str(evaluator),
        '--track-csv', str(args.track_csv),
        '--checkpoint', str(args.checkpoint),
        '--output', str(out_dir),
        '--scenario-set', 'physical',
        '--seconds', str(args.seconds),
        '--rollout-steps', str(args.rollout_steps),
        '--actor-lr', str(args.actor_lr),
        '--critic-lr', str(args.critic_lr),
        '--seed', '0', '--sampling-seed', str(seed),
        '--ablation-components', '--split-critic', '--qfpsi-signed-sweep',
    ]


def execute(args):
    evaluator = Path(__file__).resolve().with_name('adapt_nmpc.py')
    if not evaluator.is_file():
        raise FileNotFoundError(f'Cannot find evaluator: {evaluator}')
    seeds = tuple(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError('Seeds must be unique non-negative integers')
    args.checkpoint = Path(args.checkpoint).resolve(strict=True)
    args.track_csv = Path(args.track_csv).resolve(strict=True)
    if args.checkpoint.suffix.lower() != '.npz':
        raise ValueError('Expected NPZ checkpoint')
    if args.track_csv.suffix.lower() != '.csv':
        raise ValueError('Expected CAD CSV reference')
    if args.seconds <= 0 or args.rollout_steps < 1 or args.actor_lr <= 0 or args.critic_lr <= 0:
        raise ValueError('Positive simulation duration/rollout steps/learning rates required')
    root = Path(args.output_root).resolve()
    if root == evaluator.parent or root in (args.checkpoint, args.track_csv):
        raise ValueError('Output root must be independent of evaluator and inputs')
    dirs = {seed: root / f'seed_{seed}' for seed in seeds}
    if not args.summary_only and not args.dry_run:
        for seed, out in dirs.items():
            if out.exists():
                if not args.resume:
                    raise FileExistsError(
                        f'{out} exists; select new --output-root, or use '
                        '--resume for VALIDATED finished seeds / --summary-only'
                    )
                inspect_seed(out, seed, seconds=args.seconds,
                             rollout_steps=args.rollout_steps,
                             actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                             checkpoint=args.checkpoint, track_csv=args.track_csv)
                print(f'[skip verified] {out}', flush=True)
        root.mkdir(parents=True, exist_ok=True)
    elif args.dry_run:
        print('DRY RUN: no simulator execution, no files written', flush=True)

    if not args.summary_only:
        for seed, out in dirs.items():
            cmd = make_command(args, seed, out, evaluator)
            if args.dry_run:
                print(' '.join('"' + a + '"' if ' ' in a else a for a in cmd))
                continue
            if args.resume and out.exists():
                continue
            print(f'\n[run] seed={seed}; output={out}', flush=True)
            subprocess.run(cmd, cwd=evaluator.parent, check=True)
            inspect_seed(out, seed, seconds=args.seconds,
                         rollout_steps=args.rollout_steps,
                         actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                         checkpoint=args.checkpoint, track_csv=args.track_csv)
            print(f'[validated] seed={seed}, 4 scenarios x 5 offsets', flush=True)
    if args.dry_run:
        return None
    rows = []
    for seed, out in dirs.items():
        rows.extend(inspect_seed(out, seed, seconds=args.seconds,
                                 rollout_steps=args.rollout_steps,
                                 actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                                 checkpoint=args.checkpoint, track_csv=args.track_csv))
    detailed, macro = summarize_all(rows, seeds)
    summary_dir = unique_summary_dir(root)
    write_csv(summary_dir / 'phase2d_seed_scenario_offset.csv', rows)
    write_csv(summary_dir / 'phase2d_scenario_offset_summary.csv', detailed)
    write_csv(summary_dir / 'phase2d_macro_by_offset.csv', macro)
    provenance = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': sha256(args.checkpoint),
        'held_out_cad': str(args.track_csv),
        'cad_sha256': sha256(args.track_csv),
        'evaluator_sha256': sha256(evaluator),
        'seeds': list(seeds), 'scenario_set': list(SCENARIOS),
        'signed_offsets_pct': list(OFFSETS), 'seconds': args.seconds,
        'rollout_steps': args.rollout_steps,
        'actor_lr': args.actor_lr, 'critic_lr': args.critic_lr,
        'r3a_checkpoint_unchanged': True,
        'aggregation_unit': 'seed with 4 scenarios averaged within seed',
        'study_limitations': [
            'One held-out CAD geometry; the four plant cases are not four independent tracks',
            'N=5 independent Actor sampling seeds; report paired uncertainty',
            'Solver success does not imply hard real-time feasibility or closed-loop stability',
            'No terminal head learning; this is a sensitivity study, not Phase 2E',
        ],
    }
    (summary_dir / 'phase2d_provenance.json').write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print('\nPhysical-plant Phase 2D multiseed summary (paired Δ vs Qfψ 0%):')
    for row in macro:
        print(
            f"  Qfψ {row['offset_pct']:+d}%: "
            f"ΔmeanCTE={row['average_delta_mean_cte_cm']:+.5f} cm "
            f"({row['mean_cte_improved_seed_count']}/{row['n_seeds']} macro-seeds improve); "
            f"ΔmaxCTE={row['average_delta_max_cte_cm']:+.5f} cm; "
            f"Δheading={row['average_delta_heading_rmse_rad']:+.6f} rad; "
            f"Δreward={row['average_delta_reward']:+.2f}; "
            f"solverMin={row['min_solver_success_pct']:.2f}%; "
            f"deadlineMissWorst={row['worst_deadline_miss_pct']:.3f}%",
            flush=True,
        )
    print(f'\nSaved validated aggregation: {summary_dir}', flush=True)
    return summary_dir


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--track-csv', type=Path, required=True)
    p.add_argument('--output-root', type=Path, default=(
        Path(__file__).resolve().parent / '..' / '..' / 'experiment_archive'
        / 'eval_cad_split_2d_physical_multiseed'
    ))
    p.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    p.add_argument('--seconds', type=float, default=150.)
    p.add_argument('--rollout-steps', type=int, default=16)
    p.add_argument('--actor-lr', type=float, default=.001)
    p.add_argument('--critic-lr', type=float, default=.001)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--dry-run', action='store_true',
                   help='Show evaluator subprocess commands without writing files')
    p.add_argument('--resume', action='store_true',
                   help='Skip only COMPLETED and VALIDATED seed directories; reject partial dirs')
    p.add_argument('--summary-only', action='store_true',
                   help='Validate existing seeds and create a new summary; do not simulate')
    args = p.parse_args(argv)
    if args.summary_only and args.dry_run:
        p.error('--summary-only and --dry-run are mutually exclusive')
    if args.summary_only and args.resume:
        p.error('--summary-only implies no reruns; omit --resume')
    return args


if __name__ == '__main__':
    try:
        execute(parse_args())
    except (OSError, ValueError, subprocess.CalledProcessError) as ex:
        print(f'Phase 2D multiseed FAILED: {ex}', file=sys.stderr)
        sys.exit(1)
