"""Evaluate a pretrained Direct-P Actor NN / six-parameter Critic on unseen CAD geometry.

Runs a STRICT held-out analytic S-bend from adapt_nmpc.install_evaluation_track;
training source is never edited or used for evaluation. Offline "frozen" arms do
not learn on test data. Online arm is explicitly transductive/test-time adaptation.

Run from nmpc_casadi:
 python eval_direct_p_heldout.py --checkpoint direct_results/train/checkpoint.npz \
   --output direct_results/heldout --seconds 120
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import rl_nmpc as rl
from direct_p_critic import DirectPAgent
from adapt_nmpc import install_evaluation_track, plant_for_profile

CONTROLLERS = (
    'fixed_nmpc',
    'actor_frozen_initial_p_sampled',
    'direct_p_frozen_sampled',
    'direct_p_frozen_mean',
    'direct_p_online_sampled',
)
PROFILES = ('nominal', 'payload', 'tire_segment', 'rolling_segment')


def construct_agent(args, metadata):
    """All arms reload the same checkpoint and its stored RNG state."""
    initial = np.asarray(metadata['critic_initial'], dtype=float)
    p_min = np.asarray(metadata['critic_p_min'], dtype=float)
    p_max = np.asarray(metadata['critic_p_max'], dtype=float)
    low = float(p_min[0] / initial[0])
    high = float(p_max[0] / initial[0])
    agent = DirectPAgent(
        seed=args.seed,
        sigma=float(metadata['sigma']),
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        critic_nsteps=1,
        p_scale=metadata['critic_p_scale'],
        p_min_multiplier=low,
        p_max_multiplier=high,
    )
    agent.load(args.checkpoint)
    return agent


def evaluate_one(args, env, metadata, profile, controller):
    agent = construct_agent(args, metadata) if controller != 'fixed_nmpc' else None
    obs, error = env.reset(args.ey, np.deg2rad(args.heading_deg),
                           rl.b.p_nominal.copy())
    training_qf = rl.QF_INIT.copy()
    fixed_weights = tuple(v.copy() for v in env.initial_weights)
    updates = []
    infer_ms, learn_ms = [], []
    while True:
        if agent is None:
            weights = fixed_weights
            latent = None
            infer_ms.append(0.)
        else:
            tic = time.perf_counter_ns()
            weights, latent = agent.propose(
                obs, explore=controller != 'direct_p_frozen_mean')
            infer_ms.append((time.perf_counter_ns() - tic) * 1e-6)
            if controller == 'actor_frozen_initial_p_sampled':
                # Identical pretrained Actor / Gaussian noise, isolate learned P.
                weights = (weights[0], weights[1], training_qf.copy())
        env.plant = plant_for_profile(profile, env.k * rl.b.Ts, args)
        next_obs, next_error, reward, done, terminated = env.step(weights)
        env.logs[-1]['profile'] = profile
        env.logs[-1]['critic_online_update'] = int(
            controller == 'direct_p_online_sampled')
        if agent is None:
            learn_ms.append(0.)
        elif controller == 'direct_p_online_sampled':
            tic = time.perf_counter_ns()
            diag = agent.learn([(
                obs.copy(), error.copy(), np.asarray(latent).copy(),
                float(reward), next_obs.copy(), next_error.copy(),
                float(terminated)
            )])
            learn_ms.append((time.perf_counter_ns() - tic) * 1e-6)
            updates.append(dict(step=env.k, **diag))
        else:
            learn_ms.append(0.)
        obs, error = next_obs, next_error
        if done:
            break
    log = pd.DataFrame(env.logs)
    log['policy_inference_ms'] = infer_ms
    log['learning_ms'] = learn_ms
    log['loop_core_ms'] = log['solve_ms'] + log['policy_inference_ms'] + log['learning_ms']
    summary = rl.metrics(env.logs)
    summary.update(
        profile=profile, controller=controller, steps=len(log),
        retry_steps=int((log['solver_attempt_count'] > 1).sum()),
        total_updates=len(updates),
        final_p=(agent.critic.physical_diagonal().tolist() if agent is not None
                 else rl.QF_INIT.tolist()),
        core_loop_p95_ms=float(log['loop_core_ms'].quantile(.95)),
        core_loop_p99_ms=float(log['loop_core_ms'].quantile(.99)),
        core_loop_deadline_miss_pct=float(100 * (log['loop_core_ms'] > rl.b.Ts*1000).mean()),
    )
    return log, pd.DataFrame(updates), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=120.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ey', type=float, default=.05)
    parser.add_argument('--heading-deg', type=float, default=0.)
    parser.add_argument('--actor-lr', type=float, default=.001)
    parser.add_argument('--critic-lr', type=float, default=.001)
    parser.add_argument('--nominal-only', action='store_true')
    parser.add_argument('--track', choices=['heldout_s_bend', 'training'],
                        default='heldout_s_bend')
    parser.add_argument('--track-csv', type=Path, default=None)
    parser.add_argument('--payload-start', type=float, default=30.)
    parser.add_argument('--payload-ramp-s', type=float, default=10.)
    parser.add_argument('--payload-fraction', type=float, default=.10)
    parser.add_argument('--road-start', type=float, default=30.)
    parser.add_argument('--road-end', type=float, default=60.)
    parser.add_argument('--road-transition-s', type=float, default=5.)
    parser.add_argument('--cornering-drop', type=float, default=.20)
    parser.add_argument('--rolling-increase', type=float, default=.25)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be positive finite')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('--output must be an empty directory (no overwrites)')
    args.output.mkdir(parents=True, exist_ok=True)
    args.checkpoint = args.checkpoint.resolve(strict=True)
    with np.load(args.checkpoint, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata'].item()))
    if metadata.get('critic_architecture') != 'direct_diagonal_quadratic':
        parser.error('Checkpoint is NOT the six-parameter Direct-P architecture')
    construct_agent(args, metadata)  # Fail before changing any evaluation source.

    reference = install_evaluation_track(
        args.output, args.track, args.track_csv)
    if args.track == 'heldout_s_bend' and not reference['held_out']:
        raise RuntimeError('Evaluation track was not held out')
    # IMPORTANT: install the held-out reference BEFORE constructing Environment.
    env = rl.Environment(args.seconds)
    settings = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
    }
    settings['checkpoint_meta'] = metadata
    settings['held_out'] = bool(reference['held_out'])
    settings['no_heldout_checkpoint_selection'] = True
    (args.output / 'config.json').write_text(json.dumps(settings, indent=2))

    output_rows = []
    for profile in PROFILES[:1] if args.nominal_only else PROFILES:
        for controller in CONTROLLERS:
            log, updates, result = evaluate_one(
                args, env, metadata, profile, controller)
            stem = f'{profile}_{controller}'
            log.to_csv(args.output / f'{stem}.csv', index=False)
            if len(updates):
                updates.to_csv(args.output / f'{stem}_updates.csv', index=False)
            output_rows.append(result)
            pd.DataFrame(output_rows).to_csv(
                args.output / 'comparison.csv', index=False)
            readable = {k: result[k] for k in (
                'profile', 'controller', 'mean_cte_post_m', 'max_cte_post_m',
                'rmse_heading_post_rad', 'return', 'reached_end',
                'solver_success_pct', 'retry_steps', 'total_updates')}
            print(json.dumps(readable, allow_nan=True), flush=True)

    table = pd.DataFrame(output_rows)
    metrics = ('mean_cte_post_m', 'max_cte_post_m',
               'rmse_heading_post_rad', 'return')
    paired = []
    for profile in table.profile.unique():
        subset = table[table.profile == profile].set_index('controller')
        if not all(c in subset.index for c in CONTROLLERS):
            continue
        for test in CONTROLLERS[1:]:
            row = {'profile': profile, 'controller': test,
                   'baseline': 'fixed_nmpc'}
            for metric in metrics:
                row[metric+'_difference_vs_fixed'] = float(
                    subset.loc[test, metric] - subset.loc['fixed_nmpc', metric])
            # Pure learned-Critic contribution relative to the SAME Actor.
            for metric in metrics:
                row[metric+'_difference_vs_actor_fixed_p'] = float(
                    subset.loc[test, metric] -
                    subset.loc['actor_frozen_initial_p_sampled', metric])
            paired.append(row)
    pd.DataFrame(paired).to_csv(args.output / 'paired_deltas.csv', index=False)

    # A compact report summarizing the strict held-out frozen-policy arm.
    frozen = table[table.controller == 'direct_p_frozen_sampled']
    fixed = table[table.controller == 'fixed_nmpc']
    actor_pfixed = table[table.controller == 'actor_frozen_initial_p_sampled']
    print('HELDOUT_SUMMARY_JSON=' + json.dumps({
        'held_out': reference['held_out'],
        'reference_sha256': reference.get('reference_sha256'),
        'training_reference_sha256': reference.get('training_reference_sha256'),
        'profiles': frozen.profile.tolist(),
        'mean_cte_post_cm': {
            'fixed': float(fixed.mean_cte_post_m.mean()*100),
            'actor_fixed_p': float(actor_pfixed.mean_cte_post_m.mean()*100),
            'actor_and_learned_p': float(frozen.mean_cte_post_m.mean()*100)
        },
        'completed': {
            'fixed': int(fixed.reached_end.sum()),
            'actor_fixed_p': int(actor_pfixed.reached_end.sum()),
            'actor_and_learned_p': int(frozen.reached_end.sum())
        }
    }, allow_nan=True), flush=True)


if __name__ == '__main__':
    main()
