

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rl_nmpc as rl
from cad_eval_reference import prepare_cad_evaluation_reference
from ac_ablation import learn_selected_component, net_unchanged, snapshot_net, net_parameter_change_norm
from qfpsi_isolation import install_pretrained_qfpsi_function, verify_frozen_qfpsi
from heldout_reference import (
    HELDOUT_ID, HELDOUT_CURVATURE_FORMULA,
    compare_geometry, describe_geometry,
    generate_heldout_s_bend, sha256_file,
)


CONTROLLERS = ('fixed_nmpc', 'ac_nmpc_frozen', 'ac_nmpc_online')
FROZEN_DETERMINISTIC = 'ac_nmpc_frozen_deterministic'
ABLATION_MODES = {'ac_nmpc_actor_only': 'actor', 'ac_nmpc_critic_only': 'critic'}
QFPSI_HOLD_MODES = {
    'ac_nmpc_critic_qfpsi_held': 'critic',
    'ac_nmpc_online_qfpsi_held': 'both',
}
DISPLAY = {
    'fixed_nmpc': 'Fixed NMPC',
    'ac_nmpc_frozen': 'Frozen AC-NMPC (sampled)',
    'ac_nmpc_frozen_deterministic': 'Frozen AC-NMPC (mean)',
    'ac_nmpc_online': 'Online AC-NMPC',
    'ac_nmpc_actor_only': 'Online Actor-only',
    'ac_nmpc_critic_only': 'Online Critic-only',
    'ac_nmpc_critic_qfpsi_held': 'Critic-only, Qf_psi held',
    'ac_nmpc_online_qfpsi_held': 'Online both, Qf_psi held',
}


def active_controllers(include_frozen_deterministic=False, include_ablation=False,
                       isolate_qfpsi=False):
    """Default roster unchanged; optional Qf_psi(s) causal intervention."""
    if isolate_qfpsi and not include_ablation:
        raise ValueError('Qf_psi isolation requires --ablation-components')
    if not include_ablation:
        if include_frozen_deterministic:
            return ('fixed_nmpc', 'ac_nmpc_frozen',
                    FROZEN_DETERMINISTIC, 'ac_nmpc_online')
        return CONTROLLERS
    order = ['fixed_nmpc', 'ac_nmpc_frozen']
    if include_frozen_deterministic:
        order.append(FROZEN_DETERMINISTIC)
    order.extend(ABLATION_MODES)
    if isolate_qfpsi:
        order.append('ac_nmpc_critic_qfpsi_held')
    order.append('ac_nmpc_online')
    if isolate_qfpsi:
        order.append('ac_nmpc_online_qfpsi_held')
    return tuple(order)



# Capture immutable training-reference identity BEFORE the evaluation-only path
# is installed. Never alter the original source CSV or pretrained checkpoint.
TRAINING_REF_SOURCE = Path(rl.b.REF_SOURCE)
TRAINING_LENGTH_M = float(rl.b.CAD_S_END)


def install_evaluation_track(output, track, track_csv=None):
    """Switch only the running evaluator's ReferenceManager source.

    No changes to rl_nmpc.py, baseline.py, training trajectory or checkpoint.
    Every controller re-instantiates the same reference via Environment.reset.
    """
    if track_csv is not None:
        # An evaluator must not relabel a training reference as held-out.
        # Neither training CSV is modified.
        custom_source = Path(track_csv).expanduser().resolve(strict=True)
        training_candidates = [
            rl.b.BASE / 'xref_states_smoothed.csv',
            rl.b.BASE / 'xref_states_controller.csv',
        ]
        for candidate in training_candidates:
            if candidate.exists() and (
                custom_source == candidate.resolve()
                or sha256_file(custom_source) == sha256_file(candidate)
            ):
                raise ValueError(
                    'Refusing to evaluate a training CAD source with --track-csv: '
                    + str(candidate)
                )
        ref_info, info = prepare_cad_evaluation_reference(
            custom_source, Path(output), rl.b.TAIL_LENGTH_M,
            min_speed_mps=float(rl.b.cfg['vx_min']),
            max_speed_mps=min(float(rl.b.TEST_SPEED_MPS),
                              float(rl.b.cfg['vx_max'])),
            wheelbase_m=float(rl.b.WHEELBASE_M),
            steering_limit_rad=float(min(
                abs(rl.b.cfg['u_min'][1]), abs(rl.b.cfg['u_max'][1]))),
        )
        rl.b.TAIL_FILE = Path(info['tail_file'])
        rl.b.REF_INFO = ref_info
        rl.b.CAD_S_END = float(ref_info['cad_s_end'])
        info['note'] = (
            'Independent user-supplied CAD reference for evaluation only. '
            'Training and checkpoint sources remain unchanged.'
        )
    elif track == 'training':
        # Explicit legacy replay. Not an out-of-distribution generalization test.
        info = {
            'id': 'training_original',
            'held_out': False,
            'reference_file': str(TRAINING_REF_SOURCE),
            'reference_sha256': sha256_file(TRAINING_REF_SOURCE),
            'reference_length_m': float(rl.b.CAD_S_END),
            'note': 'Legacy in-distribution replay, NOT evidence of unseen-path generalization',
        }
    elif track == 'heldout_s_bend':
        path = Path(output) / f'{HELDOUT_ID}_reference.csv'
        tail_path = Path(output) / f'{HELDOUT_ID}_with_virtual_tail.csv'
        data = generate_heldout_s_bend(
            path, length_m=TRAINING_LENGTH_M,
            speed_mps=rl.b.TEST_SPEED_MPS,
        )
        diff = compare_geometry(TRAINING_REF_SOURCE, data)
        # Baseline generates a virtual horizon-only tail and records the REAL
        # path length separately. Only these runtime b.* values are switched.
        ref_info = rl.b.prepare_reference_with_tail(
            path, tail_path, rl.b.TAIL_LENGTH_M
        )
        rl.b.TAIL_FILE = tail_path
        rl.b.REF_INFO = ref_info
        rl.b.CAD_S_END = float(ref_info['cad_s_end'])
        if not np.isclose(rl.b.CAD_S_END, TRAINING_LENGTH_M, atol=1e-8):
            raise RuntimeError('Heldout geometry changed the task path length')
        info = {
            'id': HELDOUT_ID,
            'held_out': True,
            'not_used_for_training_or_checkpoint_selection': True,
            'source_geometry': 'independent analytic curvature integration',
            'curvature_formula': HELDOUT_CURVATURE_FORMULA,
            'reference_file': str(path),
            'reference_sha256': sha256_file(path),
            'tail_file': str(tail_path),
            'tail_sha256': sha256_file(tail_path),
            'geometry': describe_geometry(data),
            'difference_vs_training': diff,
        }
    else:
        raise ValueError(f'Unknown evaluation track: {track}')
    info['training_reference_file'] = str(TRAINING_REF_SOURCE)
    info['training_reference_sha256'] = sha256_file(TRAINING_REF_SOURCE)
    info['training_reference_length_m'] = TRAINING_LENGTH_M
    (Path(output) / 'reference_manifest.json').write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    return info


def fresh_agent(args, meta):
    """Reload exactly the same pretrained checkpoint for every AC-NMPC scenario."""
    kw = dict(
        seed=args.seed,
        sigma=meta.get('sigma', rl.SIGMA),
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        critic_nsteps=meta.get('critic_nsteps', 16),
    )
    if 'train_component' in inspect.signature(rl.Agent).parameters:
        kw.update(
            train_component=meta.get('train_component', 'both'),
            initial_seed=meta.get('initial_seed'),
        )
    agent = rl.Agent(**kw)
    agent.load(args.checkpoint)
    # Agent.load restores the checkpoint's RNG state, overriding --seed.
    # New opt-in evaluation seed varies Gaussian exploration without changing
    # pretrained weights/Adam moments. Every control arm gets the SAME seed.
    sampling_seed = getattr(args, 'sampling_seed', None)
    if sampling_seed is not None:
        agent.rng = np.random.default_rng(sampling_seed)
    if getattr(agent, 'train_component', 'both') != 'both':
        raise ValueError(
            'This comparison requires an Actor-Critic checkpoint with train_component="both". '
            f'Checkpoint reports {getattr(agent, "train_component", None)!r}.'
        )
    return agent


def _percentile_summary(values, prefix):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {
            f'{prefix}_mean_ms': 0.0,
            f'{prefix}_p50_ms': 0.0,
            f'{prefix}_p95_ms': 0.0,
            f'{prefix}_p99_ms': 0.0,
            f'{prefix}_max_ms': 0.0,
        }
    return {
        f'{prefix}_mean_ms': float(np.mean(a)),
        f'{prefix}_p50_ms': float(np.quantile(a, 0.50)),
        f'{prefix}_p95_ms': float(np.quantile(a, 0.95)),
        f'{prefix}_p99_ms': float(np.quantile(a, 0.99)),
        f'{prefix}_max_ms': float(np.max(a)),
    }


# Current plant parameter order: [m, Iz, Cf, Cr, lf, lr, mu_roll, g, eps_v]
P_M, P_IZ, P_CF, P_CR, P_MU_ROLL = 0, 1, 2, 3, 6
PLANT_LOG_FIELDS = {
    P_M: 'm',
    P_IZ: 'Iz',
    P_CF: 'Cf',
    P_CR: 'Cr',
    P_MU_ROLL: 'mu_roll',
}


def _smooth_ramp(t, start, duration):
    """C1 half-cosine ramp: 0 before start, 1 after start+duration."""
    if duration <= 0:
        return float(t >= start)
    x = np.clip((t - start) / duration, 0.0, 1.0)
    return float(0.5 - 0.5 * np.cos(np.pi * x))


def _smooth_window(t, start, end, transition):
    """Finite smooth window with half-cosine entry and exit."""
    if end <= start:
        raise ValueError('road-end must be greater than road-start')
    if t < start or t > end:
        return 0.0
    tr = max(0.0, min(float(transition), 0.5 * (end - start)))
    if tr == 0.0:
        return 1.0
    if t < start + tr:
        x = (t - start) / tr
        return float(0.5 - 0.5 * np.cos(np.pi * x))
    if t > end - tr:
        x = (end - t) / tr
        return float(0.5 - 0.5 * np.cos(np.pi * x))
    return 1.0


def plant_for_profile(profile, t, args):
    """Plant parameters at time t.  NMPC's internal model remains nominal."""
    p0 = rl.b.p_nominal
    if len(p0) <= P_MU_ROLL:
        raise RuntimeError(
            'Unexpected plant parameter vector: expected [m,Iz,Cf,Cr,lf,lr,mu_roll,...]'
        )
    p = p0.copy()

    load_level = _smooth_ramp(t, args.payload_start, args.payload_ramp_s)
    load_factor = 1.0 + args.payload_fraction * load_level
    road_level = _smooth_window(t, args.road_start, args.road_end, args.road_transition_s)

    if profile == 'nominal':
        pass
    elif profile == 'payload':
        # Equivalent distributed payload: keep Iz/m constant and leave CG geometry fixed.
        p[P_M] = p0[P_M] * load_factor
        p[P_IZ] = p0[P_IZ] * load_factor
    elif profile == 'mass_only':
        p[P_M] = p0[P_M] * load_factor
    elif profile == 'Iz_only':
        p[P_IZ] = p0[P_IZ] * load_factor
    elif profile == 'Cf_segment':
        p[P_CF] = p0[P_CF] * (1.0 - args.cornering_drop * road_level)
    elif profile == 'Cr_segment':
        p[P_CR] = p0[P_CR] * (1.0 - args.cornering_drop * road_level)
    elif profile == 'tire_segment':
        p[P_CF] = p0[P_CF] * (1.0 - args.cornering_drop * road_level)
        p[P_CR] = p0[P_CR] * (1.0 - args.cornering_drop * road_level)
    elif profile == 'rolling_segment':
        p[P_MU_ROLL] = p0[P_MU_ROLL] * (1.0 + args.rolling_increase * road_level)
    else:
        raise ValueError(f'Unknown plant profile: {profile}')
    return p


def build_scenarios(args):
    scenarios = [dict(name='nominal', ey=.05, heading=0.0, profile='nominal')]
    if args.nominal_only:
        return scenarios

    if args.scenario_set in ('physical', 'all'):
        scenarios += [
            dict(name='payload_ramp', ey=.05, heading=0.0, profile='payload'),
            dict(name='tire_segment', ey=.05, heading=0.0, profile='tire_segment'),
            dict(name='rolling_segment', ey=.05, heading=0.0, profile='rolling_segment'),
        ]

    if args.scenario_set in ('ofat', 'all'):
        scenarios += [
            dict(name='mass_ramp', ey=.05, heading=0.0, profile='mass_only'),
            dict(name='Iz_ramp', ey=.05, heading=0.0, profile='Iz_only'),
            dict(name='Cf_segment', ey=.05, heading=0.0, profile='Cf_segment'),
            dict(name='Cr_segment', ey=.05, heading=0.0, profile='Cr_segment'),
        ]
        if args.scenario_set == 'ofat':
            scenarios.append(
                dict(name='rolling_segment', ey=.05, heading=0.0, profile='rolling_segment')
            )

    if args.include_initial_perturbations:
        for ey, deg in [(-.05, 5), (.05, -5), (.1, -10)]:
            scenarios.append(
                dict(name=f'ey{ey}_psi{deg}', ey=ey, heading=np.deg2rad(deg), profile='nominal')
            )
    return scenarios


def _attach_plant_to_last_log(env, profile, p_plant):
    if not env.logs:
        return
    row = env.logs[-1]
    row['plant_profile'] = profile
    for idx, label in PLANT_LOG_FIELDS.items():
        row[f'plant_{label}'] = float(p_plant[idx])
        row[f'plant_{label}_ratio'] = float(p_plant[idx] / rl.b.p_nominal[idx])


def _finalize_timing(logs, propose_timing=None, learn_timing_by_step=None):
    """Add a common timing schema so the two controllers can be compared directly."""
    n = len(logs)
    if propose_timing is None:
        propose_timing = np.zeros(n, dtype=float)
    else:
        propose_timing = np.asarray(propose_timing, dtype=float)
    if len(propose_timing) != n:
        raise RuntimeError(f'Policy timing length {len(propose_timing)} != log length {n}')

    if learn_timing_by_step is None:
        learn_timing_by_step = np.zeros(n, dtype=float)
    else:
        learn_timing_by_step = np.asarray(learn_timing_by_step, dtype=float)
    if len(learn_timing_by_step) != n:
        raise RuntimeError(f'Learning timing length {len(learn_timing_by_step)} != log length {n}')

    logs['policy_inference_ms'] = propose_timing
    logs['learn_ms'] = learn_timing_by_step
    logs['env_non_solver_ms'] = logs['env_step_wall_ms'] - logs['solve_ms']
    logs['control_step_no_learn_ms'] = logs['policy_inference_ms'] + logs['env_step_wall_ms']
    logs['control_step_total_ms'] = logs['control_step_no_learn_ms'] + logs['learn_ms']
    logs['core_solver_plus_policy_ms'] = logs['policy_inference_ms'] + logs['solve_ms']
    logs['core_solver_policy_learn_ms'] = logs['core_solver_plus_policy_ms'] + logs['learn_ms']

    deadline_ms = float(rl.b.Ts * 1000.0)
    logs['deadline_ms'] = deadline_ms
    logs['deadline_miss_no_learn'] = (
        logs['control_step_no_learn_ms'] > deadline_ms
    ).astype(int)
    logs['deadline_miss_total'] = (
        logs['control_step_total_ms'] > deadline_ms
    ).astype(int)
    return logs


def _timing_metrics(logs, wall_ms, learning_call_times):
    out = {}
    for col, prefix in [
        ('policy_inference_ms', 'policy_inference'),
        ('solve_ms', 'solver'),
        ('env_non_solver_ms', 'env_non_solver'),
        ('env_step_wall_ms', 'env_step'),
        ('learn_ms', 'learning_per_step'),
        ('control_step_no_learn_ms', 'control_step_no_learn'),
        ('control_step_total_ms', 'control_step_total'),
    ]:
        out.update(_percentile_summary(logs[col].to_numpy(float), prefix))

    active = logs.loc[logs['learn_ms'] > 0, 'learn_ms'].to_numpy(float)
    out.update(_percentile_summary(active, 'learning_active'))

    deadline_ms = float(rl.b.Ts * 1000.0)
    out.update(
        deadline_ms=deadline_ms,
        deadline_miss_no_learn_pct=float(100.0 * logs['deadline_miss_no_learn'].mean()),
        deadline_miss_total_pct=float(100.0 * logs['deadline_miss_total'].mean()),
        control_step_total_margin_p99_ms=float(
            deadline_ms - np.quantile(logs['control_step_total_ms'].to_numpy(float), 0.99)
        ),
        rollout_wall_ms=float(wall_ms),
        rollout_wall_per_step_ms=float(wall_ms / max(len(logs), 1)),
        learning_total_ms=float(np.sum(learning_call_times)) if learning_call_times else 0.0,
    )
    return out


def run_fixed_nmpc(env, args, scenario):
    """Pure fixed-weight NMPC baseline: no Actor, no Critic, no network update."""
    name = scenario['name']
    ey = float(scenario['ey'])
    heading = float(scenario['heading'])
    profile = scenario['profile']

    # Reset from nominal plant; plant_for_profile() overwrites it before each propagation.
    env.reset(ey, heading, rl.b.p_nominal.copy())
    env.learning_updates = []
    fixed_weights = tuple(np.asarray(w, dtype=float).copy() for w in env.initial_weights)

    step_timing = []
    wall_start = time.perf_counter_ns()
    while True:
        t_plant = float(env.k * rl.b.Ts)
        p_plant = plant_for_profile(profile, t_plant, args)
        env.plant = p_plant.copy()

        start = time.perf_counter_ns()
        _, _, _, done, _ = env.step(fixed_weights)
        step_timing.append((time.perf_counter_ns() - start) / 1e6)
        _attach_plant_to_last_log(env, profile, p_plant)

        if done:
            break
    wall_ms = (time.perf_counter_ns() - wall_start) / 1e6

    logs = pd.DataFrame(env.logs)
    if len(logs) != len(step_timing):
        raise RuntimeError(
            f'Timing alignment failed for {name}/fixed_nmpc: logs={len(logs)}, step={len(step_timing)}'
        )
    logs['env_step_wall_ms'] = np.asarray(step_timing, dtype=float)
    logs['updates_before_action'] = 0
    logs = _finalize_timing(logs)
    logs.to_csv(args.output / f'{name}_fixed_nmpc.csv', index=False)

    m = rl.metrics(env.logs)
    m.update(
        scenario=name,
        plant_profile=profile,
        controller='fixed_nmpc',
        updates=0,
        **_timing_metrics(logs, wall_ms, []),
    )
    return m


def run_ac_nmpc_online(env, agent, args, scenario, *,
                       train_component='both', controller_label='ac_nmpc_online',
                       hold_qfpsi=False):
    """Sampled AC-NMPC with selective learning and optional functional Qf_psi hold."""
    if train_component not in ('both', 'actor', 'critic'):
        raise ValueError('Unknown online ablation mode')
    if hold_qfpsi:
        expected = next(
            (key for key, value in QFPSI_HOLD_MODES.items()
             if value == train_component), None,
        )
        if expected is None:
            raise ValueError('Qf_psi hold only supported with Critic learning')
    else:
        expected = ('ac_nmpc_online' if train_component == 'both'
                    else next(k for k, v in ABLATION_MODES.items()
                              if v == train_component))
    if controller_label != expected:
        raise ValueError('Controller label/mode mismatch')
    initial_actor = snapshot_net(agent.actor)
    initial_critic = snapshot_net(agent.critic)
    name = scenario['name']
    ey = float(scenario['ey'])
    heading = float(scenario['heading'])
    profile = scenario['profile']

    original_learn = agent.learn
    original_propose = agent.propose
    original_step = env.step

    # Qf_psi(s) is masked at the FUNCTION level using an immutable pretrained
    # critic. Freezing only its output-layer column would not suffice when
    # shared hidden layers continue to change.
    original_terms = None
    frozen_qfpsi = None
    if hold_qfpsi:
        original_terms, frozen_qfpsi = install_pretrained_qfpsi_function(agent)

    learn_call_times = []
    propose_timing = []
    step_timing = []

    def timed_learn(rows):
        start = time.perf_counter_ns()
        result = learn_selected_component(
            agent, rows, train_component, learn_fn=original_learn,
        )
        elapsed = (time.perf_counter_ns() - start) / 1e6
        learn_call_times.append(elapsed)
        return {**result, 'learn_ms': elapsed}

    def timed_propose(*a, **kw):
        start = time.perf_counter_ns()
        result = original_propose(*a, **kw)
        if hold_qfpsi:
            verify_frozen_qfpsi(result[0], float(frozen_qfpsi(a[0])))
        propose_timing.append((time.perf_counter_ns() - start) / 1e6)
        return result

    def timed_step(*a, **kw):
        # Same plant profile as fixed NMPC.  This scheduling calculation is excluded
        # from controller timing because it emulates the external physical plant.
        t_plant = float(env.k * rl.b.Ts)
        p_plant = plant_for_profile(profile, t_plant, args)
        env.plant = p_plant.copy()

        start = time.perf_counter_ns()
        result = original_step(*a, **kw)
        step_timing.append((time.perf_counter_ns() - start) / 1e6)
        _attach_plant_to_last_log(env, profile, p_plant)
        return result

    agent.learn = timed_learn
    agent.propose = timed_propose
    env.step = timed_step

    wall_start = time.perf_counter_ns()
    try:
        _, m = rl.rollout(
            env,
            agent,
            True,  # sampled actions are required for the current score-function Actor update
            ey,
            heading,
            plant=rl.b.p_nominal.copy(),
            train=True,
            rollout_steps=args.rollout_steps,
            controller='actor_critic',
        )
    finally:
        agent.learn = original_learn
        agent.propose = original_propose
        env.step = original_step
        if hold_qfpsi:
            agent.critic_terms = original_terms
    wall_ms = (time.perf_counter_ns() - wall_start) / 1e6

    logs = pd.DataFrame(env.logs)
    n = len(logs)
    if len(propose_timing) != n or len(step_timing) != n:
        raise RuntimeError(
            f'Timing alignment failed for {name}/ac_nmpc_online: '
            f'logs={n}, propose={len(propose_timing)}, step={len(step_timing)}'
        )

    logs['env_step_wall_ms'] = np.asarray(step_timing, dtype=float)
    logs['updates_before_action'] = 0
    learn_by_step = np.zeros(n, dtype=float)
    updates = getattr(env, 'learning_updates', [])
    for update in updates:
        k = int(update['step'])
        if not (1 <= k <= n):
            raise RuntimeError(f'Invalid learning-update step {k} for log length {n}')
        learn_by_step[k - 1] += float(update.get('learn_ms', 0.0))
        logs.loc[k:, 'updates_before_action'] += 1

    logs = _finalize_timing(logs, propose_timing=propose_timing, learn_timing_by_step=learn_by_step)
    logs.to_csv(args.output / f'{name}_{controller_label}.csv', index=False)

    # Preserve legacy output filenames for the original Online condition.
    if train_component == 'both':
        update_name = f'{name}_ac_updates.csv'
        checkpoint_name = f'{name}_ac_adapted_checkpoint.npz'
    else:
        update_name = f'{name}_{controller_label}_updates.csv'
        checkpoint_name = f'{name}_{controller_label}_adapted_checkpoint.npz'
    pd.DataFrame(updates).to_csv(args.output / update_name, index=False)
    # Critical: the held-head arm is a two-network inference composition.
    # Saving a plain Agent checkpoint would NOT preserve that held function,
    # so do not issue a misleading standalone adapted checkpoint.
    if not hold_qfpsi:
        agent.save(args.output / checkpoint_name)
    unchanged_actor = net_unchanged(agent.actor, initial_actor)
    unchanged_critic = net_unchanged(agent.critic, initial_critic)
    if train_component == 'actor' and not unchanged_critic:
        raise RuntimeError('Actor-only changed pretrained Critic parameters or Adam state')
    if train_component == 'critic' and not unchanged_actor:
        raise RuntimeError('Critic-only changed pretrained Actor parameters or Adam state')

    m.update(
        scenario=name,
        plant_profile=profile,
        controller=controller_label,
        train_component=train_component,
        qfpsi_function_held_at_checkpoint=bool(hold_qfpsi),
        adapted_checkpoint_saved=not bool(hold_qfpsi),
        actor_unchanged_from_checkpoint=unchanged_actor,
        critic_unchanged_from_checkpoint=unchanged_critic,
        actor_parameter_change_norm=net_parameter_change_norm(
            agent.actor, initial_actor,
        ),
        critic_parameter_change_norm=net_parameter_change_norm(
            agent.critic, initial_critic,
        ),
        updates=len(updates),
        **_timing_metrics(logs, wall_ms, learn_call_times),
    )
    return m


def snapshot_network_state(agent):
    """Exact parameters and Adam moments/step counters; RNG intentionally excluded."""
    return {
        name: {
            'p': [arr.copy() for arr in getattr(agent, name).p],
            'm': [arr.copy() for arr in getattr(agent, name).m],
            'v': [arr.copy() for arr in getattr(agent, name).v],
            't': getattr(agent, name).t,
        }
        for name in ('actor', 'critic')
    }


def network_state_unchanged(agent, saved):
    """Frozen means that both policy parameters and optimizer states are unchanged."""
    for name in ('actor', 'critic'):
        net = getattr(agent, name)
        old = saved[name]
        if net.t != old['t']:
            return False
        for attr in ('p', 'm', 'v'):
            arrays, previous = getattr(net, attr), old[attr]
            if len(arrays) != len(previous):
                return False
            if any(not np.array_equal(x, y) for x, y in zip(arrays, previous)):
                return False
    return True


def run_ac_nmpc_frozen(env, agent, args, scenario, deterministic=False):
    """Run the same AC-NMPC feedback policy without a single network update.

    Stochastic Frozen samples from agent.propose(s, explore=True), exactly as
    Online does. Freshly loaded agents use the same RNG state; they draw the
    same Gaussian noise per time step until their trajectories/weights diverge.
    Deterministic Frozen (optional) uses explore=False.
    """
    name = scenario['name']
    ey = float(scenario['ey'])
    heading = float(scenario['heading'])
    profile = scenario['profile']
    label = FROZEN_DETERMINISTIC if deterministic else 'ac_nmpc_frozen'

    saved = snapshot_network_state(agent)
    propose_timing = []
    step_timing = []
    wall_start = time.perf_counter_ns()
    s, _ = env.reset(ey, heading, rl.b.p_nominal.copy())
    env.learning_updates = []
    while True:
        start = time.perf_counter_ns()
        weights, _ = agent.propose(s, explore=not deterministic)
        propose_timing.append((time.perf_counter_ns() - start) / 1e6)

        # Same time-varying physical plant and nominal NMPC predictor as Online.
        t_plant = float(env.k * rl.b.Ts)
        p_plant = plant_for_profile(profile, t_plant, args)
        env.plant = p_plant.copy()

        start = time.perf_counter_ns()
        sn, _, _, done, _ = env.step(weights)
        step_timing.append((time.perf_counter_ns() - start) / 1e6)
        _attach_plant_to_last_log(env, profile, p_plant)
        s = sn
        if done:
            break
    wall_ms = (time.perf_counter_ns() - wall_start) / 1e6

    # Never call agent.learn, including at episode end. Assert all parameters
    # AND optimizer states are bitwise unchanged.
    if getattr(env, 'learning_updates', []):
        raise RuntimeError(f'Frozen run unexpectedly updated networks: {name}')
    if not network_state_unchanged(agent, saved):
        raise RuntimeError(f'Frozen Actor/Critic or Adam state changed: {name}')

    logs = pd.DataFrame(env.logs)
    n = len(logs)
    if len(propose_timing) != n or len(step_timing) != n:
        raise RuntimeError(
            f'Timing alignment failed for {name}/{label}: '
            f'logs={n}, propose={len(propose_timing)}, step={len(step_timing)}'
        )
    logs['env_step_wall_ms'] = np.asarray(step_timing, dtype=float)
    logs['updates_before_action'] = 0
    logs = _finalize_timing(logs, propose_timing=propose_timing)
    logs.to_csv(args.output / f'{name}_{label}.csv', index=False)

    m = rl.metrics(env.logs)
    m.update(
        scenario=name, plant_profile=profile, controller=label, updates=0,
        frozen_network_unchanged=True,
        sampling_mode='deterministic_mean' if deterministic else 'sampled_gaussian',
        **_timing_metrics(logs, wall_ms, []),
    )
    return m


def _improve_lower(baseline, candidate, metric):
    """Positive improvement percentage means candidate is better."""
    bv = float(baseline[metric])
    cv = float(candidate[metric])
    if not np.isfinite(bv) or not np.isfinite(cv) or abs(bv) < 1e-12:
        return np.nan
    return 100.0 * (bv - cv) / abs(bv)


def build_pairwise_summary(comparison):
    """One row per scenario, retaining historical Fixed-vs-Online columns."""
    rows = []
    for scenario, group in comparison.groupby('scenario', sort=False):
        idx = group.set_index('controller')
        if not set(CONTROLLERS).issubset(idx.index):
            continue
        f, z, a = [idx.loc[name] for name in CONTROLLERS]
        rows.append({
            'scenario': scenario,
            'plant_profile': f['plant_profile'],
            'fixed_tracking_feasible': bool(f['tracking_feasible']),
            'frozen_tracking_feasible': bool(z['tracking_feasible']),
            'ac_tracking_feasible': bool(a['tracking_feasible']),
            'fixed_mean_cte_post_m': float(f['mean_cte_post_m']),
            'frozen_mean_cte_post_m': float(z['mean_cte_post_m']),
            'ac_mean_cte_post_m': float(a['mean_cte_post_m']),
            'mean_cte_post_improvement_pct': _improve_lower(f, a, 'mean_cte_post_m'),
            'frozen_vs_fixed_mean_cte_improvement_pct':
                _improve_lower(f, z, 'mean_cte_post_m'),
            'online_vs_frozen_mean_cte_improvement_pct':
                _improve_lower(z, a, 'mean_cte_post_m'),
            'fixed_max_cte_post_m': float(f['max_cte_post_m']),
            'frozen_max_cte_post_m': float(z['max_cte_post_m']),
            'ac_max_cte_post_m': float(a['max_cte_post_m']),
            'max_cte_post_improvement_pct': _improve_lower(f, a, 'max_cte_post_m'),
            'online_vs_frozen_max_cte_improvement_pct':
                _improve_lower(z, a, 'max_cte_post_m'),
            'fixed_heading_rmse_post_rad': float(f['rmse_heading_post_rad']),
            'frozen_heading_rmse_post_rad': float(z['rmse_heading_post_rad']),
            'ac_heading_rmse_post_rad': float(a['rmse_heading_post_rad']),
            'heading_rmse_post_improvement_pct':
                _improve_lower(f, a, 'rmse_heading_post_rad'),
            'online_vs_frozen_heading_rmse_improvement_pct':
                _improve_lower(z, a, 'rmse_heading_post_rad'),
            'fixed_vx_rmse_post_mps': float(f['rmse_vx_post_mps']),
            'frozen_vx_rmse_post_mps': float(z['rmse_vx_post_mps']),
            'ac_vx_rmse_post_mps': float(a['rmse_vx_post_mps']),
            'vx_rmse_post_improvement_pct': _improve_lower(f, a, 'rmse_vx_post_mps'),
            'online_vs_frozen_vx_rmse_improvement_pct':
                _improve_lower(z, a, 'rmse_vx_post_mps'),
            'fixed_solver_success_pct': float(f['solver_success_pct']),
            'frozen_solver_success_pct': float(z['solver_success_pct']),
            'ac_solver_success_pct': float(a['solver_success_pct']),
            'solver_success_delta_pctpoint': float(
                a['solver_success_pct'] - f['solver_success_pct']),
            'fixed_completion_pct': float(f['completion_pct']),
            'frozen_completion_pct': float(z['completion_pct']),
            'ac_completion_pct': float(a['completion_pct']),
            'completion_delta_pctpoint': float(
                a['completion_pct'] - f['completion_pct']),
            'fixed_return': float(f['return']),
            'frozen_return': float(z['return']),
            'ac_return': float(a['return']),
            'return_delta': float(a['return'] - f['return']),
            'online_vs_frozen_return_delta': float(a['return'] - z['return']),
            'fixed_control_step_mean_ms': float(f['control_step_total_mean_ms']),
            'frozen_control_step_mean_ms': float(z['control_step_total_mean_ms']),
            'ac_control_step_mean_ms': float(a['control_step_total_mean_ms']),
            'ac_timing_overhead_mean_ms': float(
                a['control_step_total_mean_ms'] - f['control_step_total_mean_ms']),
            'online_vs_frozen_timing_overhead_mean_ms': float(
                a['control_step_total_mean_ms'] - z['control_step_total_mean_ms']),
            'fixed_control_step_p99_ms': float(f['control_step_total_p99_ms']),
            'frozen_control_step_p99_ms': float(z['control_step_total_p99_ms']),
            'ac_control_step_p99_ms': float(a['control_step_total_p99_ms']),
        })
    return pd.DataFrame(rows)


def build_pairwise_long(comparison):
    """Explicit pairwise contrasts including the frozen-learning effect."""
    pairs = (
        ('frozen_vs_fixed', 'ac_nmpc_frozen', 'fixed_nmpc'),
        ('online_vs_frozen', 'ac_nmpc_online', 'ac_nmpc_frozen'),
        ('online_vs_fixed', 'ac_nmpc_online', 'fixed_nmpc'),
        ('frozen_mean_vs_sampled', FROZEN_DETERMINISTIC, 'ac_nmpc_frozen'),
        ('actor_only_vs_frozen', 'ac_nmpc_actor_only', 'ac_nmpc_frozen'),
        ('critic_only_vs_frozen', 'ac_nmpc_critic_only', 'ac_nmpc_frozen'),
        ('both_vs_actor_only', 'ac_nmpc_online', 'ac_nmpc_actor_only'),
        ('both_vs_critic_only', 'ac_nmpc_online', 'ac_nmpc_critic_only'),
        ('qfpsi_hold_critic_vs_critic', 'ac_nmpc_critic_qfpsi_held',
         'ac_nmpc_critic_only'),
        ('qfpsi_hold_both_vs_both', 'ac_nmpc_online_qfpsi_held',
         'ac_nmpc_online'),
        ('qfpsi_hold_critic_vs_frozen', 'ac_nmpc_critic_qfpsi_held',
         'ac_nmpc_frozen'),
        ('qfpsi_hold_both_vs_frozen', 'ac_nmpc_online_qfpsi_held',
         'ac_nmpc_frozen'),
    )
    rows = []
    for scenario, group in comparison.groupby('scenario', sort=False):
        idx = group.set_index('controller')
        for name, candidate, baseline in pairs:
            if candidate not in idx.index or baseline not in idx.index:
                continue
            a, b = idx.loc[candidate], idx.loc[baseline]
            entry = {
                'scenario': scenario,
                'comparison': name,
                'candidate': candidate,
                'baseline': baseline,
                'candidate_updates': int(a['updates']),
                'baseline_updates': int(b['updates']),
                'candidate_reached_end': bool(a['reached_end']),
                'baseline_reached_end': bool(b['reached_end']),
                'candidate_tracking_feasible': bool(a['tracking_feasible']),
                'baseline_tracking_feasible': bool(b['tracking_feasible']),
            }
            for metric in ('mean_cte_post_m', 'max_cte_post_m',
                           'rmse_heading_post_rad', 'rmse_vx_post_mps',
                           'mean_du_delta_sq'):
                entry[metric + '_improvement_pct'] = _improve_lower(b, a, metric)
                entry[metric + '_delta'] = float(a[metric] - b[metric])
            entry['return_delta'] = float(a['return'] - b['return'])
            entry['completion_pct_delta'] = float(a['completion_pct'] - b['completion_pct'])
            entry['control_step_mean_ms_delta'] = float(
                a['control_step_total_mean_ms'] - b['control_step_total_mean_ms'])
            rows.append(entry)
    return pd.DataFrame(rows)


def initial_sampled_pairing(output, scenario, rollout_steps):
    """Check common action prefix before the first Online network update.

    Both modes load the same checkpoint, use Gaussian sampling, and see the
    same plant. Initial actions should agree until online's first batch update.
    Report mismatches rather than silently assuming paired samples.
    """
    f = pd.read_csv(Path(output)/f'{scenario}_ac_nmpc_frozen.csv')
    a = pd.read_csv(Path(output)/f'{scenario}_ac_nmpc_online.csv')
    n = min(len(f), len(a), int(rollout_steps))
    columns = [c for c in f.columns if c.startswith(('q_', 'r_', 'qf_'))
               and c in a.columns]
    if n == 0 or not columns:
        raise RuntimeError(f'No comparable Frozen/Online weights in {scenario}')
    fw = f.loc[:n-1, columns].to_numpy(float)
    aw = a.loc[:n-1, columns].to_numpy(float)
    max_weight_difference = float(np.max(np.abs(fw - aw)))
    max_cte_difference = float(np.max(np.abs(
        f['cte_m'].to_numpy(float)[:n] -
        a['cte_m'].to_numpy(float)[:n]
    )))
    return {
        'scenario':scenario,
        'pre_update_steps':int(n),
        'weight_columns_compared':len(columns),
        'max_abs_weight_difference':max_weight_difference,
        'max_abs_cte_difference_m':max_cte_difference,
        'matched_weight_prefix':bool(np.allclose(fw, aw, rtol=1e-9, atol=1e-8)),
        'matched_cte_prefix':bool(np.allclose(
            f['cte_m'].to_numpy(float)[:n],
            a['cte_m'].to_numpy(float)[:n],
            rtol=1e-9, atol=1e-8
        )),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--rollout-steps', type=int, default=16)
    ap.add_argument('--actor-lr', type=float, default=.003)
    ap.add_argument('--critic-lr', type=float, default=.001)
    ap.add_argument('--seconds', type=float, default=120)
    ap.add_argument('--seed', type=int, default=0,
                    help='Legacy Agent constructor seed; checkpoint load restores RNG')
    ap.add_argument('--sampling-seed', type=int, default=None,
                    help='Optional independent Gaussian exploration seed AFTER checkpoint load; '
                         'omit to reproduce legacy experiments exactly')
    track_selection = ap.add_mutually_exclusive_group()
    track_selection.add_argument(
        '--track', choices=['heldout_s_bend', 'training'],
        default='heldout_s_bend',
        help='Default: unseen S-bend; training is an explicit legacy replay only',
    )
    track_selection.add_argument(
        '--track-csv', type=Path,
        help='Evaluate with a NEW AutoCAD XREF CSV; training references are protected',
    )
    ap.add_argument('--nominal-only', action='store_true')
    ap.add_argument('--include-frozen-deterministic', action='store_true',
                    help='Also evaluate frozen Actor mean without Gaussian exploration')
    ap.add_argument('--ablation-components', action='store_true',
                    help='Add Actor-only and Critic-only Online control arms')
    ap.add_argument('--isolate-qfpsi', action='store_true',
                    help='Also compare Critic-only and both Online with pretrained '
                         'Qf_psi(s) held fixed while other Critic outputs adapt')
    ap.add_argument(
        '--scenario-set', choices=['physical', 'ofat', 'all'], default='physical',
        help='physical: grouped physical cases; ofat: one factor at a time; all: both',
    )
    ap.add_argument('--include-initial-perturbations', action='store_true')

    # Payload/load profile.
    ap.add_argument('--payload-start', type=float, default=30.0)
    ap.add_argument('--payload-ramp-s', type=float, default=10.0)
    ap.add_argument('--payload-fraction', type=float, default=0.10)

    # Tire/road finite segment.
    ap.add_argument('--road-start', type=float, default=30.0)
    ap.add_argument('--road-end', type=float, default=60.0)
    ap.add_argument('--road-transition-s', type=float, default=5.0)
    ap.add_argument('--cornering-drop', type=float, default=0.20)
    ap.add_argument('--rolling-increase', type=float, default=0.25)
    args = ap.parse_args()

    if args.rollout_steps < 1 or args.seconds <= 0 or not np.isfinite(args.seconds):
        ap.error('rollout-steps and seconds must be positive')
    if args.sampling_seed is not None and args.sampling_seed < 0:
        ap.error('--sampling-seed must be a non-negative integer')
    if args.isolate_qfpsi and not args.ablation_components:
        ap.error('--isolate-qfpsi requires --ablation-components')
    for lr in [args.actor_lr, args.critic_lr]:
        if not np.isfinite(lr) or lr <= 0:
            ap.error('learning rates must be positive and finite')
    if args.payload_start < 0 or args.payload_ramp_s < 0:
        ap.error('payload-start and payload-ramp-s must be non-negative')
    if args.road_start < 0 or args.road_end <= args.road_start or args.road_transition_s < 0:
        ap.error('Require 0 <= road-start < road-end and road-transition-s >= 0')
    for name, value in [
        ('payload-fraction', args.payload_fraction),
        ('cornering-drop', args.cornering_drop),
        ('rolling-increase', args.rolling_increase),
    ]:
        if not np.isfinite(value) or value < 0:
            ap.error(f'{name} must be finite and non-negative')
    if args.cornering_drop >= 1.0:
        ap.error('cornering-drop must be < 1 so Cf/Cr stay positive')

    with np.load(args.checkpoint, allow_pickle=False) as saved:
        meta = json.loads(str(saved['metadata'].item()))

    # Validate that this really is an Actor-Critic checkpoint before making outputs.
    fresh_agent(args, meta)

    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        ap.error('Use an empty output folder to preserve prior results')

    reference_metadata = install_evaluation_track(args.output, args.track, args.track_csv)
    controllers = active_controllers(
        args.include_frozen_deterministic,
        include_ablation=args.ablation_components,
        isolate_qfpsi=args.isolate_qfpsi,
    )
    env = rl.Environment(args.seconds)
    fixed_q, fixed_r, fixed_qf = [np.asarray(w, dtype=float).tolist() for w in env.initial_weights]

    config = dict(
        mode='fixed_vs_frozen_vs_online_actor_critic_nmpc',
        evaluation_track=reference_metadata['id'],
        evaluation_reference=reference_metadata,
        evaluation_path_is_held_out=bool(reference_metadata['held_out']),
        checkpoint_source=str(args.checkpoint),
        checkpoint_metadata=meta,
        controllers=list(controllers),
        comparison_definition={
            'fixed_nmpc': (
                'Pure NMPC baseline. Q/R/Qf = Environment.initial_weights for every sample. '
                'No Actor, no Critic, no neural-network inference, no online learning.'
            ),
            'ac_nmpc_frozen': (
                'Same pretrained checkpoint as Online, same per-step Gaussian sampling and '
                'state-dependent Q/R/Qf, but Actor/Critic and Adam optimizer are never updated.'
            ),
            'ac_nmpc_frozen_deterministic': (
                'Optional frozen inference using Actor mean, without Gaussian sampling and '
                'without any gradient/network update.'
            ),
            'ac_nmpc_online': (
                'Same pretrained checkpoint and Gaussian sampling as stochastic Frozen. '
                'Actor/Critic infer Q/R/Qf per step and both networks update online through agent.learn().'
            ),
            'ac_nmpc_actor_only': (
                'Only Actor updates Q/R; Critic/Qf network and Adam state stay frozen. '
                'Both gradients are still computed: timing is diagnostic only.'
            ),
            'ac_nmpc_critic_only': (
                'Only Critic updates Qf/value; Actor/Q/R network and Adam state stay frozen. '
                'Both gradients are still computed: timing is diagnostic only.'
            ),
            'ac_nmpc_critic_qfpsi_held': (
                'Only Critic updates, but applied Qf_psi(s) and its value-term '
                'derivative are held at the pretrained Critic FUNCTION. '
                'All other Critic outputs adapt; diagnostic-only composition.'
            ),
            'ac_nmpc_online_qfpsi_held': (
                'Actor+Critic update, but applied Qf_psi(s) is held at pretrained '
                'Critic function. Other Critic outputs and Actor continue updating; '
                'no standalone adapted checkpoint is produced.'
            ),
        },
        fixed_nmpc_weights={'Q': fixed_q, 'R': fixed_r, 'Qf': fixed_qf},
        fairness={
            'same_initial_condition_per_scenario': True,
            'same_time_varying_plant_profile': True,
            'same_nmpc_internal_model': 'b.p_nominal',
            'checkpoint_reset_for_each_ac_scenario_and_controller': True,
            'frozen_online_same_sampled_rng_initial_state': True,
            'same_sampling_seed_for_every_actor_critic_arm': True,
            'frozen_online_same_explore_true': True,
            'frozen_has_zero_learning_updates': True,
            'online_rollout_steps': args.rollout_steps,
            'same_reference_geometry_per_controller': True,
            'heldout_path_not_seen_in_training': bool(reference_metadata['held_out']),
            'independent_evaluation_reference_from_file': args.track_csv is not None,
        },
        ablation_components=bool(args.ablation_components),
        isolate_qfpsi=bool(args.isolate_qfpsi),
        rollout_steps=args.rollout_steps,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        seconds=args.seconds,
        seed=args.seed,
        sampling_seed=args.sampling_seed,
        gaussian_sampling_rng_policy=(
            'explicit_independent_seed_after_checkpoint_load'
            if args.sampling_seed is not None
            else 'checkpoint_restored_rng_state'
        ),
        Ts_s=float(rl.b.Ts),
        control_deadline_ms=float(rl.b.Ts * 1000.0),
        scenario_set=args.scenario_set,
        plant_parameter_order=['m', 'Iz', 'Cf', 'Cr', 'lf', 'lr', 'mu_roll', 'g', 'eps_v'],
        plant_mismatch={
            'controller_model': 'All controllers keep NMPC model b.p_nominal.',
            'payload': {
                'start_s': args.payload_start,
                'ramp_s': args.payload_ramp_s,
                'fractional_increase': args.payload_fraction,
                'assumption': 'm and Iz scale together so Iz/m is constant; CG, lf and lr unchanged',
            },
            'tire_segment': {
                'start_s': args.road_start,
                'end_s': args.road_end,
                'transition_s': args.road_transition_s,
                'effective_cornering_stiffness_drop': args.cornering_drop,
            },
            'rolling_segment': {
                'start_s': args.road_start,
                'end_s': args.road_end,
                'transition_s': args.road_transition_s,
                'rolling_resistance_increase': args.rolling_increase,
                'note': 'mu_roll is the coefficient used in -mu*g, not tire-road friction mu',
            },
        },
        timing_note=(
            'Fixed NMPC has policy_inference_ms=0 and learn_ms=0. '
            'Frozen AC-NMPC has state-dependent inference, but learn_ms=0. '
            'Online AC-NMPC has both inference and learning. '
            'Environment.step wall time includes simulated plant propagation; '
            'these are Python simulation-process timings rather than embedded WCET. '
            'For sampled Frozen vs Online, common Gaussian draws are seeded identically; '
            'their observations/policies diverge only after online updates.'
        ),
    )
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))

    scenarios = build_scenarios(args)
    results = []
    sampled_pairing = []

    for scenario in scenarios:
        for controller in controllers:
            if controller == 'fixed_nmpc':
                m = run_fixed_nmpc(env, args, scenario)
            elif controller == 'ac_nmpc_frozen':
                m = run_ac_nmpc_frozen(
                    env, fresh_agent(args, meta), args, scenario, deterministic=False
                )
            elif controller == FROZEN_DETERMINISTIC:
                m = run_ac_nmpc_frozen(
                    env, fresh_agent(args, meta), args, scenario, deterministic=True
                )
            elif controller == 'ac_nmpc_online':
                m = run_ac_nmpc_online(
                    env, fresh_agent(args, meta), args, scenario
                )
            elif controller in ABLATION_MODES:
                m = run_ac_nmpc_online(
                    env, fresh_agent(args, meta), args, scenario,
                    train_component=ABLATION_MODES[controller],
                    controller_label=controller,
                )
            elif controller in QFPSI_HOLD_MODES:
                m = run_ac_nmpc_online(
                    env, fresh_agent(args, meta), args, scenario,
                    train_component=QFPSI_HOLD_MODES[controller],
                    controller_label=controller,
                    hold_qfpsi=True,
                )
            else:
                raise ValueError(f'Unknown controller: {controller}')
            m['reference_track'] = reference_metadata['id']
            results.append(m)
            pd.DataFrame(results).to_csv(args.output / 'comparison.csv', index=False)
            print(json.dumps(m), flush=True)
        sampled_pairing.append(
            initial_sampled_pairing(args.output, scenario['name'], args.rollout_steps)
        )
        pd.DataFrame(sampled_pairing).to_csv(
            args.output / 'initial_sampled_pairing.csv', index=False
        )

    comparison = pd.DataFrame(results)
    if args.ablation_components:
        # Pair ALL Online arms against the same sampled Frozen prefix. Before
        # step 16 no gradient was applied, so first actions/weights/CTE
        # must match exactly under common checkpoint/seed and plant.
        pairing_rows = []
        qfpsi_rows = []
        targets = ['ac_nmpc_frozen', 'ac_nmpc_actor_only',
                   'ac_nmpc_critic_only', 'ac_nmpc_online']
        if args.isolate_qfpsi:
            targets.extend(QFPSI_HOLD_MODES)
        for scenario in scenarios:
            name = scenario['name']
            frozen = pd.read_csv(args.output / f'{name}_ac_nmpc_frozen.csv')
            weight_cols = [col for col in frozen.columns
                           if col.startswith(('q_', 'r_', 'qf_'))]
            for controller in targets:
                d = pd.read_csv(args.output / f'{name}_{controller}.csv')
                measured = d.loc[d['progress_m'] >= 1.3]
                if measured.empty:
                    measured = d
                match_row = comparison.loc[
                    (comparison.scenario == name) &
                    (comparison.controller == controller)
                ].iloc[0]
                qfpsi_rows.append({
                    'scenario': name, 'controller': controller,
                    'updates': int(match_row['updates']),
                    'mean_cte_post_cm': 100.0 * float(match_row['mean_cte_post_m']),
                    'max_cte_post_cm': 100.0 * float(match_row['max_cte_post_m']),
                    'heading_rmse_post_rad': float(match_row['rmse_heading_post_rad']),
                    'return': float(match_row['return']),
                    'qf_psi_mean_post': float(measured['qf_2'].mean()),
                    'qf_psi_first': float(d['qf_2'].iloc[0]),
                    'qf_psi_last': float(d['qf_2'].iloc[-1]),
                    'qf_psi_max_post': float(measured['qf_2'].max()),
                    'q_psi_mean_post': float(measured['q_2'].mean()),
                    'r_delta_mean_post': float(measured['r_1'].mean()),
                })
                if controller == 'ac_nmpc_frozen':
                    continue
                n = min(len(d), len(frozen), args.rollout_steps)
                common_cols = [v for v in weight_cols if v in d.columns]
                if n < 1 or len(common_cols) != 14:
                    raise RuntimeError('Missing complete 14-weight ablation pairing log')
                wdelta = float(np.max(np.abs(
                    d[common_cols].to_numpy(float)[:n] -
                    frozen[common_cols].to_numpy(float)[:n]
                )))
                cdelta = float(np.max(np.abs(
                    d['cte_m'].to_numpy(float)[:n] -
                    frozen['cte_m'].to_numpy(float)[:n]
                )))
                pairing_rows.append({
                    'scenario': name, 'controller': controller,
                    'pre_update_steps': n,
                    'max_abs_weight_difference': wdelta,
                    'max_abs_cte_difference_m': cdelta,
                    'matched_weight_prefix': wdelta <= 1e-8,
                    'matched_cte_prefix': cdelta <= 1e-8,
                })
                if wdelta > 1e-8 or cdelta > 1e-8:
                    raise RuntimeError(
                        f'Unfair sampled prefix: scenario={name}, '
                        f'controller={controller}, weight={wdelta}, cte={cdelta}'
                    )
        pd.DataFrame(qfpsi_rows).to_csv(
            args.output / 'ablation_qfpsi_summary.csv', index=False,
        )
        pd.DataFrame(pairing_rows).to_csv(
            args.output / 'ablation_sampled_pairing.csv', index=False,
        )
    pairwise = build_pairwise_summary(comparison)
    pairwise.to_csv(args.output / 'pairwise_comparison.csv', index=False)
    build_pairwise_long(comparison).to_csv(
        args.output / 'pairwise_learning_effect.csv', index=False
    )

    timing_cols = [
        c for c in comparison.columns
        if c in ['scenario', 'plant_profile', 'controller', 'updates']
        or c.startswith('policy_inference_')
        or c.startswith('solver_')
        or c.startswith('env_non_solver_')
        or c.startswith('env_step_')
        or c.startswith('learning_')
        or c.startswith('control_step_')
        or c.startswith('deadline_')
        or c.startswith('rollout_wall_')
    ]
    comparison[timing_cols].to_csv(args.output / 'timing_summary.csv', index=False)

    import matplotlib.pyplot as plt

    # Same trajectory panel: fixed, frozen sampled, online (+ optional frozen mean).
    for scenario in scenarios:
        name = scenario['name']
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(
            rl.b.REF_INFO['cad_X'], rl.b.REF_INFO['cad_Y'],
            'k--', linewidth=2, label='Reference',
        )
        for controller in controllers:
            d = pd.read_csv(args.output / f'{name}_{controller}.csv')
            line, = ax.plot(d.X, d.Y, linewidth=1.6, label=DISPLAY[controller])
            color = line.get_color()
            ax.scatter(d.X.iloc[0], d.Y.iloc[0], color=color, marker='o', s=30)
            ax.scatter(d.X.iloc[-1], d.Y.iloc[-1], color=color, marker='x', s=60)
        ax.set(xlabel='X [m]', ylabel='Y [m]', title=f'{name}: Fixed / Frozen / Online AC-NMPC')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output / f'{name}_trajectory.png', dpi=180)
        plt.close(fig)

    # Nominal tracking comparison.
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for controller in controllers:
        d = pd.read_csv(args.output / f'nominal_{controller}.csv')
        axes[0].plot(d.t_s, d.cte_m * 100.0, label=DISPLAY[controller])
        axes[1].plot(d.t_s, np.rad2deg(d.e_psi), label=DISPLAY[controller])
        axes[2].plot(d.t_s, d.progress_m / rl.b.CAD_S_END * 100.0, label=DISPLAY[controller])
    axes[0].set_ylabel('CTE [cm]')
    axes[1].set_ylabel('Heading error [deg]')
    axes[2].set_ylabel('Completion [%]')
    axes[2].set_xlabel('Time [s]')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / 'nominal_fixed_vs_ac.png', dpi=170)
    plt.close(fig)

    # AC-NMPC adaptive weights in nominal operation.
    d = pd.read_csv(args.output / 'nominal_ac_nmpc_online.csv')
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, prefix in zip(axes, ['q', 'r', 'qf']):
        for col in d.columns:
            if col.startswith(prefix + '_'):
                ax.plot(d.t_s, d[col], label=col)
        ax.set_ylabel(prefix)
        ax.legend(ncol=3)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('Time [s]')
    fig.tight_layout()
    fig.savefig(args.output / 'ac_nmpc_online_weights_nominal.png', dpi=170)
    plt.close(fig)

    # Mismatch response: same plant on top, direct fixed-vs-AC tracking in the middle,
    # and AC adaptive multipliers at the bottom.
    for scenario in scenarios:
        if scenario['profile'] == 'nominal':
            continue
        name = scenario['name']
        ac = pd.read_csv(args.output / f'{name}_ac_nmpc_online.csv')
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

        changed = []
        for label in ['m', 'Iz', 'Cf', 'Cr', 'mu_roll']:
            col = f'plant_{label}_ratio'
            if col in ac and np.max(np.abs(ac[col].to_numpy(float) - 1.0)) > 1e-8:
                axes[0].plot(ac.t_s, ac[col], label=label)
                changed.append(label)
        axes[0].axhline(1.0, linestyle='--', linewidth=1.0)
        axes[0].set_ylabel('Plant / nominal')
        axes[0].set_title(f'{name}: same plant mismatch, Fixed / Frozen / Online AC-NMPC')
        if changed:
            axes[0].legend(ncol=len(changed))

        for controller in controllers:
            d = pd.read_csv(args.output / f'{name}_{controller}.csv')
            axes[1].plot(d.t_s, d.cte_m * 100.0, label=DISPLAY[controller])
            axes[2].plot(d.t_s, np.rad2deg(d.e_psi), label=DISPLAY[controller])
        axes[1].set_ylabel('CTE [cm]')
        axes[2].set_ylabel('Heading error [deg]')
        axes[1].legend(ncol=min(len(controllers), 4))
        axes[2].legend(ncol=min(len(controllers), 4))

        qbase, rbase, fbase = rl.Q_BASE, rl.R_BASE, rl.F_BASE
        if 'q_4' in ac:
            axes[3].plot(ac.t_s, ac.q_4 / qbase[4], label='Actor alpha_pos')
        if 'q_2' in ac:
            axes[3].plot(ac.t_s, ac.q_2 / qbase[2], label='Actor alpha_psi')
        if 'r_1' in ac:
            axes[3].plot(ac.t_s, ac.r_1 / rbase[1], label='Actor beta_delta')
        if 'qf_4' in ac:
            axes[3].plot(ac.t_s, ac.qf_4 / fbase[4], label='Critic m_Qf,pos')
        if 'qf_2' in ac:
            axes[3].plot(ac.t_s, ac.qf_2 / fbase[2], label='Critic m_Qf,psi')
        axes[3].set_ylabel('AC multiplier')
        axes[3].set_xlabel('Time [s]')
        axes[3].legend(ncol=3)

        for ax in axes:
            ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(args.output / f'{name}_fixed_vs_ac_response.png', dpi=170)
        plt.close(fig)

    # Direct computational-cost comparison.
    fig, ax = plt.subplots(figsize=(11, 5))
    for controller in controllers:
        d = pd.read_csv(args.output / f'nominal_{controller}.csv')
        ax.plot(d.t_s, d.control_step_total_ms, linewidth=1.0, label=DISPLAY[controller])
    ax.axhline(
        rl.b.Ts * 1000.0, linestyle='--', linewidth=1.2,
        label=f'deadline = {rl.b.Ts * 1000:.1f} ms',
    )
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Processing time / step [ms]')
    ax.set_title('Nominal computational cost: Fixed / Frozen / Online AC-NMPC')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / 'timing_fixed_vs_ac.png', dpi=170)
    plt.close(fig)

    # Compact comparison across all control modes (rather than only fixed/online).
    if not pairwise.empty:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8))
        names = pairwise['scenario'].tolist()
        x = np.arange(len(names))
        width = 0.78 / len(controllers)
        for j, controller in enumerate(controllers):
            group = comparison.loc[comparison.controller == controller].set_index('scenario')
            group = group.reindex(names)
            offset = (j - (len(controllers)-1)/2.0)*width
            axes[0].bar(x+offset, group['mean_cte_post_m'].to_numpy(float)*100.0,
                        width, label=DISPLAY[controller])
            axes[1].bar(x+offset, group['max_cte_post_m'].to_numpy(float)*100.0,
                        width, label=DISPLAY[controller])
        for ax, ylabel in zip(axes, ['Mean post CTE [cm]', 'Max post CTE [cm]']):
            ax.set_ylabel(ylabel)
            ax.set_xticks(x, names, rotation=20)
            ax.grid(axis='y', alpha=.25)
            ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(args.output / 'tracking_metric_comparison.png', dpi=170)
        plt.close(fig)

    # Direct nominal parameter-multiplier comparison: Frozen vs Online.
    frozen = pd.read_csv(args.output / 'nominal_ac_nmpc_frozen.csv')
    online = pd.read_csv(args.output / 'nominal_ac_nmpc_online.csv')
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for ax, col, nominal, title in zip(
        axes,
        ('q_4', 'q_2', 'r_1', 'qf_5'),
        (rl.Q_BASE[4], rl.Q_BASE[2], rl.R_BASE[1], rl.F_BASE[5]),
        ('Actor alpha_pos', 'Actor alpha_psi',
         'Actor beta_delta', 'Critic Qf_Y multiplier')
    ):
        ax.plot(frozen.t_s, frozen[col]/nominal,
                label=DISPLAY['ac_nmpc_frozen'])
        ax.plot(online.t_s, online[col]/nominal,
                label=DISPLAY['ac_nmpc_online'])
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=.25)
    axes[-1].set_xlabel('Time [s]')
    fig.tight_layout()
    fig.savefig(args.output / 'nominal_frozen_vs_online_multipliers.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
