"""Optional iid Gaussian six-state measurement noise for RL-NMPC simulation.

An illustrative measurement-layer model, not a calibrated sensor specification.
State order [vx m/s, vy m/s, yaw rad, yaw_rate rad/s, X m, Y m].
Noise at control index k depends only on (seed, k), so different modes
receive the identical disturbance sequence regardless of rollout length.
The physical plant never consumes these noisy measurements as ground truth.
"""
from __future__ import annotations

import numpy as np

STATE_NAMES = ("vx", "vy", "psi", "r", "X", "Y")
EXAMPLE_STD = (0.005, 0.005, float(np.deg2rad(0.5)), 0.01, 0.005, 0.005)


class GaussianStateSensor:
    def __init__(self, standard_deviations=EXAMPLE_STD, seed=0):
        std = np.asarray(standard_deviations, dtype=float)
        if std.shape != (6,) or not np.isfinite(std).all() or np.any(std < 0):
            raise ValueError("Sensor std must be six finite nonnegative numbers")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("Sensor seed must be a nonnegative integer")
        self.std = std.copy()
        self.seed = int(seed)

    def noise_at(self, step):
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer)) or step < 0:
            raise ValueError("Control step must be a nonnegative integer")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(step)]))
        return rng.normal(size=6) * self.std

    def measure(self, true_state, step):
        truth = np.asarray(true_state, dtype=float)
        if truth.shape != (6,) or not np.isfinite(truth).all():
            raise ValueError("true_state must be six finite numbers")
        noise = self.noise_at(step)
        return truth.copy() + noise, noise
