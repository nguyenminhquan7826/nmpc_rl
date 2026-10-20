"""Gaussian 17->64->64 Actor: eight independent Q/R weights.

User approved: 17 fixed-scale observations; two tanh 64-unit hidden layers;
learnable 8-dimensional Gaussian mean and dispersion heads. The Critic is
separate and contains exactly six directly learned physical P coefficients.

Paper Eq. (8): score-function policy gradient on sampled latent actions;
no differentiation through nonlinear MPC. Numerical Q/R and sigma bounds
are REQUIRED constructor inputs, not unapproved training defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

FEATURE_NAMES = (
    "vx", "vy", "psi_relative_initial", "r",
    "X_relative_origin", "Y_relative_origin",
    "e_vx", "e_vy", "e_psi", "e_r", "e_X", "e_Y",
    "a_previous", "steer_previous", "path_fraction",
    "curvature_now", "curvature_horizon_end",
)
WEIGHT_NAMES = ("q_vx", "q_vy", "q_psi", "q_r", "q_X", "q_Y",
                "r_a", "r_delta")
INPUT_DIM = 17
WEIGHT_DIM = 8
WIDTH = 64
LOG_2PI = float(np.log(2. * np.pi))


def _vec(value, n, name):
    v = np.asarray(value, dtype=float)
    if v.shape != (n,) or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be a finite vector of length {n}")
    return v.copy()


def _scalar(value, name):
    v = np.asarray(value, dtype=float)
    if v.shape != () or not np.isfinite(v):
        raise ValueError(f"{name} must be a finite scalar")
    return float(v)


def _sigmoid(z):
    return 0.5 * (1. + np.tanh(0.5 * z))


def build_actor_observation(
    state, reference_state, previous_control, *,
    path_progress_m, path_start_m, path_end_m,
    curvature_now, curvature_horizon_end,
    reference_origin_xy, reference_initial_heading_rad, train_path_length_m
):
    """Seventeen user-approved observations, fixed physical scale.

    [physical state(6), tracking error(6), previous input(2),
     path progress(1), current/preview curvature(2)].
    Origin and initial heading come from start of each reference path.
    X,Y divided by TRAIN path length on both TRAIN and held-out TEST.
    Only heading ERROR is wrapped. Physical heading must be continuous,
    e.g. the existing simulator's yaw state (do not wrap it each step).
    No TEST-fitted normalization statistics or hidden clipping.
    """
    x = _vec(state, 6, "state")
    xr = _vec(reference_state, 6, "reference_state")
    u = _vec(previous_control, 2, "previous_control")
    xy0 = _vec(reference_origin_xy, 2, "reference_origin_xy")
    h0 = _scalar(reference_initial_heading_rad, "initial reference heading")
    ltrain = _scalar(train_path_length_m, "train path length")
    s = _scalar(path_progress_m, "progress")
    ss = _scalar(path_start_m, "path start")
    se = _scalar(path_end_m, "path end")
    k0 = _scalar(curvature_now, "curvature now")
    kN = _scalar(curvature_horizon_end, "curvature preview")
    if ltrain <= 0 or se <= ss:
        raise ValueError("Both train and run path lengths must be positive")
    error = x - xr
    error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
    physical = x.copy()
    physical[2] -= h0
    physical[4:6] -= xy0
    z = np.concatenate((
        physical / np.array([.1, .06, np.pi, .5, ltrain, ltrain]),
        error / np.array([.1, .06, .2, .5, .1, .1]),
        u / np.array([2., .55]),
        np.array([(s - ss) / (se - ss), k0 / 2., kN / 2.]),
    ))
    assert z.shape == (INPUT_DIM,)
    if not np.all(np.isfinite(z)):
        raise ValueError("Non-finite Actor observation")
    return z


@dataclass(frozen=True)
class ActorProposal:
    latent: np.ndarray
    q_diagonal: np.ndarray
    r_diagonal: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    latent_log_probability: float


class GaussianWeightActor:
    """Analytically differentiable Gaussian policy with exactly 6352 params.

    Initialization centers deterministic output on the caller's nominal
    fixed NMPC Q/R; no alpha/sigma/weight limits are implicitly selected.
    This module computes gradients only; choice of optimizer and learning
    rate belongs to a future separately approved training driver.
    """

    def __init__(self, rng, nominal_q, nominal_r, q_lower, q_upper,
                 r_lower, r_upper, *, sigma_lower, sigma_upper, sigma_initial):
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy Generator")
        initial = np.r_[_vec(nominal_q, 6, "nominal Q"),
                        _vec(nominal_r, 2, "nominal R")]
        lower = np.r_[_vec(q_lower, 6, "lower Q"),
                      _vec(r_lower, 2, "lower R")]
        upper = np.r_[_vec(q_upper, 6, "upper Q"),
                      _vec(r_upper, 2, "upper R")]
        if (np.any(lower <= 0) or np.any(upper <= lower) or
                np.any(initial <= lower) or np.any(initial >= upper)):
            raise ValueError("require 0 < lower < nominal < upper for all 8 weights")
        slo = _scalar(sigma_lower, "sigma_lower")
        shi = _scalar(sigma_upper, "sigma_upper")
        sini = _scalar(sigma_initial, "sigma_initial")
        if not 0 < slo < sini < shi:
            raise ValueError("require 0 < sigma_lower < sigma_initial < sigma_upper")

        self.lower, self.upper = lower, upper
        self.sigma_lower, self.sigma_upper = slo, shi
        self.rng = rng

        def xavier(n, m):
            lim = np.sqrt(6. / (n + m))
            return rng.uniform(-lim, lim, (n, m))

        fraction = (initial - lower) / (upper - lower)
        mu0 = np.arctanh(2. * fraction - 1.)
        sigfraction = (sini - slo) / (shi - slo)
        rawsig0 = np.log(sigfraction) - np.log1p(-sigfraction)
        if not np.all(np.isfinite(mu0)) or not np.isfinite(rawsig0):
            raise ValueError("Cannot represent initial policy in supplied bounds")
        self.params = {
            "W1": xavier(INPUT_DIM, WIDTH), "b1": np.zeros(WIDTH),
            "W2": xavier(WIDTH, WIDTH), "b2": np.zeros(WIDTH),
            "W_mu": np.zeros((WIDTH, WEIGHT_DIM)), "b_mu": mu0,
            "W_sigma": np.zeros((WIDTH, WEIGHT_DIM)),
            "b_sigma": np.full(WEIGHT_DIM, rawsig0),
        }

    @property
    def parameter_count(self):
        return sum(p.size for p in self.params.values())

    def _batch(self, obs):
        a = np.asarray(obs, dtype=float)
        single = a.ndim == 1
        if single:
            a = a[None, :]
        if a.ndim != 2 or a.shape[1] != INPUT_DIM or not np.all(np.isfinite(a)):
            raise ValueError("Actor observation must be finite shape (17,) or (n,17)")
        return a, single

    def _forward(self, obs):
        p = self.params
        h1 = np.tanh(obs @ p["W1"] + p["b1"])
        h2 = np.tanh(h1 @ p["W2"] + p["b2"])
        mu = h2 @ p["W_mu"] + p["b_mu"]
        raw = h2 @ p["W_sigma"] + p["b_sigma"]
        fraction = _sigmoid(raw)
        sigma = self.sigma_lower + (self.sigma_upper - self.sigma_lower) * fraction
        return mu, sigma, (h1, h2, fraction)

    def forward(self, observation):
        obs, single = self._batch(observation)
        mu, sigma, _ = self._forward(obs)
        return (mu[0].copy(), sigma[0].copy()) if single else (mu, sigma)

    def physical_diagonals(self, latent):
        z = _vec(latent, WEIGHT_DIM, "latent action")
        w = self.lower + .5 * (self.upper - self.lower) * (1. + np.tanh(z))
        return w[:6].copy(), w[6:].copy()

    @staticmethod
    def _logp(z, mu, sigma):
        return -.5 * np.sum(((z - mu) / sigma)**2 + 2.*np.log(sigma) +
                               LOG_2PI, axis=-1)

    def latent_log_probability(self, observation, latent):
        obs, single = self._batch(observation)
        z = np.asarray(latent, dtype=float)
        if single and z.shape == (WEIGHT_DIM,):
            z = z[None, :]
        if z.shape != (len(obs), WEIGHT_DIM) or not np.all(np.isfinite(z)):
            raise ValueError("Gaussian latent shape must match observations")
        mu, sigma, _ = self._forward(obs)
        lp = self._logp(z, mu, sigma)
        return float(lp[0]) if single else lp

    def propose(self, observation, *, deterministic=False):
        obs, single = self._batch(observation)
        if not single:
            raise ValueError("One observation required for propose")
        mu, sigma, _ = self._forward(obs)
        mu, sigma = mu[0], sigma[0]
        z = mu.copy() if deterministic else mu + sigma * self.rng.standard_normal(8)
        q, r = self.physical_diagonals(z)
        return ActorProposal(z.copy(), q, r, mu.copy(), sigma.copy(),
                             float(self._logp(z[None,:],
                                              mu[None,:], sigma[None,:])[0]))

    def loss_and_gradients(self, observations, sampled_latents, frozen_td):
        """Policy-gradient DESCENT loss = -mean(stop_grad(TD)*logpi(z|s)).

        Stored Gaussian z and TD are treated as constants. The chain rule
        updates both mean and sigma heads and all shared hidden parameters;
        gradients never pass through squash/NMPC/reparameterized sampling.
        """
        obs, _ = self._batch(observations)
        z = np.asarray(sampled_latents, dtype=float)
        td = np.asarray(frozen_td, dtype=float)
        n = len(obs)
        if (n == 0 or z.shape != (n, 8) or td.shape != (n,) or
                not np.all(np.isfinite(z)) or not np.all(np.isfinite(td))):
            raise ValueError("Expected obs (n,17), latent (n,8), TD (n,)")
        mu, sigma, (h1, h2, frac) = self._forward(obs)
        loss = -float(np.mean(td * self._logp(z, mu, sigma)))
        d = z - mu
        inv = 1. / sigma
        dmu = -(td[:,None] / n) * d * inv**2
        dlogp_dsigma = (d**2 * inv**2 - 1.) * inv
        draw = -(td[:,None] / n) * dlogp_dsigma * (
            (self.sigma_upper - self.sigma_lower) * frac * (1. - frac))
        p = self.params
        grad = {
            "W_mu": h2.T @ dmu, "b_mu": np.sum(dmu, axis=0),
            "W_sigma": h2.T @ draw, "b_sigma": np.sum(draw, axis=0),
        }
        dh2 = (dmu @ p["W_mu"].T + draw @ p["W_sigma"].T) * (1.-h2**2)
        grad["W2"] = h1.T @ dh2
        grad["b2"] = np.sum(dh2, axis=0)
        dh1 = (dh2 @ p["W2"].T) * (1.-h1**2)
        grad["W1"] = obs.T @ dh1
        grad["b1"] = np.sum(dh1, axis=0)
        if not np.isfinite(loss) or any(
            not np.all(np.isfinite(g)) for g in grad.values()
        ):
            raise FloatingPointError("Nonfinite Gaussian policy gradient")
        return loss, grad
