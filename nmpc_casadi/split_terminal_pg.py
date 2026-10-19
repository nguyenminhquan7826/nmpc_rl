"""Phase 2B experimental Qf_psi control-performance policy-gradient head.

A completely separate six-output terminal MLP is initialized from the
pretrained Critic. Its HIDDEN features and other five output columns
remain FIXED. Only the Qf_psi output column/bias is trained from a
one-step, reward-derived Gaussian score-function (REINFORCE) estimator.
The TD-Huber Value Critic and original Actor continue to train via
unchanged Agent.learn() in adapt_nmpc.py.

Two paired arms use the SAME stochastic Qf_psi sampling and independent
Qf RNG: one WITHOUT terminal learning (sampling control) and the other
WITH terminal learning. Hard per-state clipping of APPLIED Qf_psi to
(1 +/- eps) of pretrained Qf_psi is a heuristic trust region and
does NOT constitute a closed-loop stability guarantee.

The Gaussian is sampled in Qf_psi RAW LOGIT space. The likelihood
gradient is calculated for the *unclipped latent sample*. This is
essential: the score-function identity remains valid for a clipped
downstream solver action, whereas differentiating through clipping or
pretending the clipped action has a Gaussian density would be wrong.
"""
from copy import deepcopy

import numpy as np
import rl_nmpc as rl
from split_critic_eval import SplitValueTerminalPolicy


PSI = 2


class TerminalPsiScorePolicy(SplitValueTerminalPolicy):
    def __init__(self, agent, *, qf_seed, sigma, lr, trust_frac,
                 trust_reg, terminal_learning):
        super().__init__(agent)
        self.sigma = float(sigma)
        self.lr = float(lr)
        self.trust_frac = float(trust_frac)
        self.trust_reg = float(trust_reg)
        self.terminal_learning = bool(terminal_learning)
        if not (np.isfinite(self.sigma) and 0. < self.sigma <= 1.):
            raise ValueError('terminal logit sampling sigma must be in (0,1]')
        if not (np.isfinite(self.lr) and 0. < self.lr <= .01):
            raise ValueError('terminal learning rate must be in (0,0.01]')
        if not (np.isfinite(self.trust_frac)
                and 0. < self.trust_frac <= .25):
            raise ValueError('terminal trust fraction must be in (0,0.25]')
        if not (np.isfinite(self.trust_reg) and 0. <= self.trust_reg <= 100.):
            raise ValueError('terminal trust regularization must be in [0,100]')
        if int(qf_seed) != qf_seed or qf_seed < 0:
            raise ValueError('qf_seed must be a nonnegative integer')
        self.qf_rng = np.random.default_rng(int(qf_seed))
        self.sample_audit = []
        self.sample_states = []
        self.pending = []
        self.qf_training_updates = 0

        # The pretrained critic's Adam moments encode OLD TD training;
        # carrying them into an unrelated policy-gradient objective causes
        # momentum-driven parameter drift even with zero new gradient.
        # Start a FRESH terminal optimizer for both paired Qf arms.
        self.terminal_head.m = [np.zeros_like(a) for a in self.terminal_head.p]
        self.terminal_head.v = [np.zeros_like(a) for a in self.terminal_head.p]
        self.terminal_head.t = 0
        self.initial_terminal = deepcopy(self.terminal_head)

    @staticmethod
    def _weight_from_logit(logit):
        return rl.F_BASE[PSI] * (
            rl.FLO + (rl.FHI - rl.FLO) * rl.sigmoid(logit)
        )

    def propose(self, s, explore):
        # Uses exactly the same Actor sampling RNG as the ordinary modes.
        (q, r, _), actor_z = super().propose(s, explore)
        raw, _ = self.terminal_head.forward(s)
        frozen_raw, _ = self.initial_terminal.forward(s)
        raw = np.asarray(raw, dtype=float)
        frozen_raw = np.asarray(frozen_raw, dtype=float)
        if raw.shape != (6,) or frozen_raw.shape != (6,):
            raise ValueError('Terminal head expected six raw outputs')
        mu_psi = float(raw[PSI])
        sample_psi = (mu_psi + float(self.qf_rng.normal(0., self.sigma))
                      if explore else mu_psi)

        ref_f = rl.F_BASE * (
            rl.FLO + (rl.FHI - rl.FLO) * rl.sigmoid(frozen_raw)
        )
        candidate_psi = float(self._weight_from_logit(sample_psi))
        low = float(ref_f[PSI] * (1. - self.trust_frac))
        high = float(ref_f[PSI] * (1. + self.trust_frac))
        applied_psi = float(np.clip(candidate_psi, low, high))
        applied = ref_f.copy()
        applied[PSI] = applied_psi

        if (not np.all(np.isfinite(applied)) or np.any(applied <= 0.) or
                not low - 1e-8 <= applied_psi <= high + 1e-8):
            raise RuntimeError('Terminal control weight is invalid/outside trust band')
        self.sample_audit.append({
            'qfpg_pretrained_psi': float(ref_f[PSI]),
            'qfpg_mean_psi': float(self._weight_from_logit(mu_psi)),
            'qfpg_candidate_psi': candidate_psi,
            'qfpg_applied_psi': applied_psi,
            'qfpg_raw_logit_mu': mu_psi,
            'qfpg_raw_logit_sample': sample_psi,
            'qfpg_applied_ratio': applied_psi / ref_f[PSI],
            'qfpg_candidate_ratio': candidate_psi / ref_f[PSI],
            'qfpg_clipped': int(not np.isclose(
                candidate_psi, applied_psi, rtol=1e-12, atol=1e-9,
            )),
        })
        state = np.asarray(s, dtype=float).copy()
        self.sample_states.append(state)
        self.pending.append((state, sample_psi, mu_psi))
        return (q, r, applied), actor_z

    def update_terminal(self, rows):
        """Update terminal head using independent one-step reward estimator.

        Must run BEFORE Agent.learn(rows) so both networks use the same
        pre-update Value Critic for bootstrapping. The final reward in each
        transition includes solver failure and terminal penalties.
        """
        n = len(rows)
        if n < 1 or len(self.pending) != n:
            raise RuntimeError('Terminal sampling/learning batch misaligned')
        s, e, actor_z, reward, sn, en, terminated = map(
            np.asarray, zip(*rows),
        )
        states = np.asarray([x[0] for x in self.pending])
        samples = np.asarray([x[1] for x in self.pending], dtype=float)
        stored_mu = np.asarray([x[2] for x in self.pending], dtype=float)
        if not np.array_equal(s, states):
            raise RuntimeError('Terminal sample state differs from rollout state')
        raw, _ = self.terminal_head.forward(s)
        mu = np.asarray(raw[:, PSI], dtype=float)
        if not np.allclose(mu, stored_mu, rtol=1e-12, atol=1e-12):
            raise RuntimeError('Terminal head changed within an unprocessed batch')

        # Stop gradient at Value Critic. This uses the same reward, terminal
        # masking, and one-step TD signal as the existing Actor advantage.
        v = np.asarray(self.agent.value(s, e), dtype=float)
        next_v = np.asarray(self.agent.value(sn, en), dtype=float)
        td = (reward + rl.GAMMA * (1. - terminated) * next_v - v)
        normalized_adv = td / max(float(np.std(td)), 1.)

        log_prob = (-.5 * ((samples - mu) / self.sigma)**2
                    - np.log(self.sigma) - .5 * np.log(2. * np.pi))
        pg_loss = float(np.mean(-normalized_adv * log_prob))
        grad_raw = np.zeros((n, 6))
        grad_raw[:, PSI] = (
            -normalized_adv * (samples - mu) / (self.sigma**2)
        ) / n

        frozen_raw, _ = self.initial_terminal.forward(s)
        frozen_weight = self._weight_from_logit(frozen_raw[:, PSI])
        mean_weight = self._weight_from_logit(mu)
        log_ratio = np.log(mean_weight / frozen_weight)
        reg_loss = float(np.mean(log_ratio**2))
        sig = rl.sigmoid(mu)
        dweight = rl.F_BASE[PSI] * (
            rl.FHI - rl.FLO
        ) * sig * (1. - sig)
        reg_grad = (
            2. * self.trust_reg * log_ratio * dweight
            / mean_weight / n
        )
        grad_raw[:, PSI] += reg_grad

        grad_stats = {'grad_norm': 0., 'clip_factor': 1.,
                      'parameter_delta': 0.}
        if self.terminal_learning:
            grads = self.terminal_head.grads(s, grad_raw)
            # Freeze feature extractor and non-heading output columns,
            # including all their Adam moments. This avoids shared-feature
            # leakage into the other five terminal weights.
            for g in grads[:-2]:
                g.fill(0.)
            grads[-2][:, [0, 1, 3, 4, 5]] = 0.
            grads[-1][[0, 1, 3, 4, 5]] = 0.
            grad_stats = self.terminal_head.update(grads, self.lr)
            self.qf_training_updates += 1
        self.pending.clear()

        return {
            'qfpg_training_enabled': float(self.terminal_learning),
            'qfpg_updates': float(self.qf_training_updates),
            'qfpg_pg_loss': pg_loss,
            'qfpg_trust_log_loss': reg_loss,
            'qfpg_control_td_mean': float(np.mean(td)),
            'qfpg_control_td_rms': float(np.sqrt(np.mean(td**2))),
            'qfpg_sampled_logit_sigma': self.sigma,
            'qfpg_trust_reg': self.trust_reg,
            'qfpg_grad_norm': float(grad_stats['grad_norm']),
            'qfpg_parameter_delta': float(grad_stats['parameter_delta']),
        }

    def inactive_weights_unchanged(self):
        """Check hidden layers AND non-heading output neurons stayed bitwise frozen."""
        initial = self.initial_terminal
        head = self.terminal_head
        for attr in ('p', 'm', 'v'):
            before = getattr(initial, attr)
            after = getattr(head, attr)
            if len(before) != len(after):
                return False
            for i in range(len(before) - 2):
                if not np.array_equal(before[i], after[i]):
                    return False
            if not np.array_equal(before[-2][:, [0, 1, 3, 4, 5]],
                                  after[-2][:, [0, 1, 3, 4, 5]]):
                return False
            if not np.array_equal(before[-1][[0, 1, 3, 4, 5]],
                                  after[-1][[0, 1, 3, 4, 5]]):
                return False
        return True

    def initial_function_unchanged_for_other_five(self, states):
        raw_new, _ = self.terminal_head.forward(states)
        raw_old, _ = self.initial_terminal.forward(states)
        return np.array_equal(
            raw_new[..., [0, 1, 3, 4, 5]],
            raw_old[..., [0, 1, 3, 4, 5]],
        )
