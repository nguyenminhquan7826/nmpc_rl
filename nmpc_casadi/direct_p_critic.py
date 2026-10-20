"""Direct diagonal quadratic TD Critic for the R3A Actor–Critic–NMPC experiment.

No neural Critic outputs: P=diag(p) is learned directly by projected TD(0)
semi-gradient. P is stored in normalized error coordinates; physical NMPC
Qf = diag(p / scale**2). Default scale is the physical error scale.

Keeps rl_nmpc.Agent's neural Actor, reward, environment, and checkpoint
machinery, while replacing only the Critic and its learning rule.
"""
import numpy as np
import rl_nmpc as rl


DEFAULT_ERROR_SCALE = np.array([.1, .06, .2, .5, .1, .1], dtype=float)


class DiagonalTDCritic:
    """Six trainable coefficients (not a neural network)."""

    def __init__(self, physical_p, scale=DEFAULT_ERROR_SCALE,
                 min_multiplier=.5, max_multiplier=50.):
        self.scale = np.asarray(scale, dtype=float).copy()
        initial = np.asarray(physical_p, dtype=float)
        if initial.shape != (6,) or self.scale.shape != (6,):
            raise ValueError('P and error scale must have six entries')
        if not np.all(np.isfinite(initial)) or not np.all(initial > 0):
            raise ValueError('Initial terminal diagonal must be positive finite')
        if not np.all(np.isfinite(self.scale)) or not np.all(self.scale > 0):
            raise ValueError('Error scale must be positive finite')
        if not (np.isfinite(min_multiplier) and np.isfinite(max_multiplier)
                and 0 < min_multiplier < 1 < max_multiplier):
            raise ValueError('Require 0 < min_multiplier < 1 < max_multiplier')
        self.p_initial = initial * self.scale**2
        self.lower = min_multiplier * self.p_initial
        self.upper = max_multiplier * self.p_initial
        # .p/.m/.v/.t maintain compatibility with checkpoint and ablation code.
        # Only .p[0] is optimized; .m and .v remain zero (projected SGD, not Adam).
        self.p = [self.p_initial.copy()]
        self.m = [np.zeros(6)]
        self.v = [np.zeros(6)]
        self.t = 0

    def features(self, error):
        e = np.asarray(error, dtype=float)
        if e.shape[-1] != 6 or not np.all(np.isfinite(e)):
            raise ValueError('Expected finite errors with trailing dimension 6')
        return (e / self.scale)**2

    def value(self, error):
        return -np.sum(self.features(error) * self.p[0], axis=-1)

    def physical_diagonal(self):
        return self.p[0] / self.scale**2

    def td_gradient(self, error, reward, next_error, terminated, gamma):
        """All bootstrap values use the SAME pre-update P. Semi-gradient only."""
        f = np.atleast_2d(self.features(error))
        fn = np.atleast_2d(self.features(next_error))
        r = np.asarray(reward, dtype=float).reshape(-1)
        d = np.asarray(terminated, dtype=float).reshape(-1)
        if (f.shape != fn.shape or len(f) != len(r) or len(f) != len(d)
                or not len(f) or not np.all(np.isfinite(r))
                or not np.all((d == 0) | (d == 1))
                or not (0 <= gamma <= 1)):
            raise ValueError('Invalid TD batch')
        v = -(f @ self.p[0])
        vn = -(fn @ self.p[0])
        target = r + gamma * (1. - d) * vn
        delta = target - v
        # d (0.5 * (target - V)^2)/dp_i = delta * (e_i/scale_i)^2.
        gradient = np.mean(delta[:, None] * f, axis=0)
        return delta, target, v, gradient

    def update(self, grads, lr):
        if len(grads) != 1:
            raise ValueError('Direct critic requires one gradient vector')
        g = np.asarray(grads[0], dtype=float)
        if g.shape != (6,) or not np.all(np.isfinite(g)):
            raise ValueError('Invalid six-dimensional Critic gradient')
        if not (np.isfinite(lr) and lr > 0):
            raise ValueError('Learning rate must be positive finite')
        old = self.p[0].copy()
        proposal = old - lr * g
        if not np.all(np.isfinite(proposal)):
            raise FloatingPointError('Nonfinite Critic update')
        self.p[0] = np.clip(proposal, self.lower, self.upper)
        self.t += 1
        return {
            'grad_norm': float(np.linalg.norm(g)),
            'clip_factor': float(np.linalg.norm(self.p[0] - old) /
                                 max(lr * float(np.linalg.norm(g)), 1e-12)),
            'parameter_delta': float(np.linalg.norm(self.p[0] - old)),
        }


class DirectPAgent(rl.Agent):
    """Same stochastic neural Actor; Critic is exactly six projected TD weights."""

    def __init__(self, seed=0, sigma=rl.SIGMA, actor_lr=1e-3,
                 critic_lr=1e-3, critic_nsteps=1, p_scale='normalized',
                 p_min_multiplier=.5, p_max_multiplier=50.):
        if critic_nsteps != 1:
            raise ValueError('Direct-P v1 deliberately implements TD(0) only')
        super().__init__(seed=seed, sigma=sigma, actor_lr=actor_lr,
                         critic_lr=critic_lr, critic_nsteps=1)
        if p_scale not in ('normalized', 'physical'):
            raise ValueError('p_scale must be normalized or physical')
        self.p_scale = p_scale
        scale = DEFAULT_ERROR_SCALE if p_scale == 'normalized' else np.ones(6)
        self.critic = DiagonalTDCritic(rl.F_BASE, scale,
                                      p_min_multiplier, p_max_multiplier)
        self.train_component = 'both'

    def critic_terms(self, s):
        """Legacy diagnostic interface; no state-dependent P and no b(s)."""
        state = np.asarray(s)
        shape = state.shape[:-1]
        mult = np.broadcast_to(self.critic.physical_diagonal()/rl.F_BASE,
                               shape + (6,)).copy()
        zero6 = np.zeros_like(mult)
        base = np.zeros(shape, dtype=float)
        raw = np.zeros(shape + (7,), dtype=float)
        return mult, zero6, base, raw

    def propose(self, s, explore):
        mu, _ = self.actor.forward(s)
        z = mu + (self.rng.normal(0., self.sigma, 3) if explore else 0.)
        alpha = rl.LO + (rl.HI - rl.LO) * (np.tanh(z) + 1.) / 2.
        q = rl.Q_BASE.copy()
        q[2] *= alpha[1]
        q[4:] *= alpha[0]
        r = rl.R_BASE.copy()
        r[1] *= alpha[2]
        return (q, r, self.critic.physical_diagonal().copy()), z

    def value(self, s, e):
        return self.critic.value(e)

    def learn(self, rows):
        s, e, z, reward, sn, en, terminated = map(np.asarray, zip(*rows))
        batch_n = len(s)
        td, target, value, gradient = self.critic.td_gradient(
            e, reward, en, terminated, rl.GAMMA)
        # Freeze advantage before any parameter updates, as in legacy R3A.
        actor_td = td.copy()
        gc = self.critic.update([gradient], self.critic_lr)

        mu, _ = self.actor.forward(s)
        advantage = actor_td / max(float(np.std(actor_td)), 1.)
        dmu_pg = -advantage[:, None] * (z - mu) / self.sigma**2 / batch_n
        tanh_mu = np.tanh(mu)
        mean_mult = rl.LO + (rl.HI-rl.LO) * (tanh_mu + 1.) / 2.
        dmult = .5 * (rl.HI-rl.LO) * (1. - tanh_mu**2)
        safe_mult = np.maximum(mean_mult, 1e-12)
        reg_cost = np.sum(rl.ACTOR_REG * np.log(safe_mult)**2, axis=1)
        dmu_reg = (2 * rl.ACTOR_REG * np.log(safe_mult) /
                   safe_mult * dmult) / batch_n
        ga = self.actor.update(self.actor.grads(s, dmu_pg + dmu_reg),
                               self.actor_lr)
        logp = -.5 * np.sum(((z-mu)/self.sigma)**2
                              + 2*np.log(self.sigma)
                              + np.log(2*np.pi), axis=1)
        pg_loss = float(np.mean(-advantage * logp))
        td_loss = float(np.mean(td**2) / 2)
        physical_p = self.critic.physical_diagonal()

        # rl.rollout aggregates numeric fields and expects baseline/horizon keys.
        return {
            'critic_target_horizon_mean': 1., 'critic_target_horizon_min': 1.,
            'critic_target_horizon_max': 1., 'critic_full_nstep_pct': 100.,
            'actor_td_mean': float(actor_td.mean()),
            'actor_td_rms': float(np.sqrt(np.mean(actor_td**2))),
            'td_mean': float(td.mean()),
            'td_rms': float(np.sqrt(np.mean(td**2))),
            'critic_loss': td_loss, 'critic_td_loss': td_loss,
            'critic_td_mse_loss': td_loss, 'critic_loss_total': td_loss,
            'actor_loss': pg_loss, 'actor_pg_loss': pg_loss,
            'actor_reg_loss': float(np.mean(reg_cost)),
            'actor_loss_total': pg_loss + float(np.mean(reg_cost)),
            'value_mean': float(value.mean()),
            'target_mean': float(target.mean()),
            'baseline_b_mean': 0., 'baseline_b_max': 0.,
            'baseline_share_of_value_pct': 0.,
            'p_update_boundary_count': float(np.sum(
                (self.critic.p[0] <= self.critic.lower + 1e-10) |
                (self.critic.p[0] >= self.critic.upper - 1e-10))),
            **{f'qf_{i}': float(v) for i, v in enumerate(physical_p)},
            **{f'qf_multiplier_{i}_mean': float(physical_p[i]/rl.F_BASE[i])
               for i in range(6)},
            **{'actor_'+key: val for key,val in ga.items()},
            **{'critic_'+key: val for key,val in gc.items()},
        }

    def _meta(self):
        meta = super()._meta()
        for key in ('critic_baseline', 'critic_td_loss_kind', 'critic_huber_delta',
                    'qf_regularizer', 'QF_REG', 'critic_bounds'):
            meta.pop(key, None)
        meta.update(version=12, critic_architecture='direct_diagonal_quadratic',
                    critic_value='-e.T @ Qf @ e', critic_td_loss='one_step_mse',
                    critic_nsteps=1, critic_optimizer='projected_sgd',
                    critic_p_scale=self.p_scale,
                    critic_error_scale=self.critic.scale.tolist(),
                    critic_p_min=self.critic.lower.tolist(),
                    critic_p_max=self.critic.upper.tolist(),
                    critic_initial=self.critic.p_initial.tolist(),
                    critic_parameter_count=6)
        return meta

    def load(self, path, sigma_override=None):
        super().load(path, sigma_override=sigma_override)
        p = self.critic.p[0]
        if np.any(p < self.critic.lower) or np.any(p > self.critic.upper):
            raise ValueError('Checkpoint P outside certified-by-bounds region')
