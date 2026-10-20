"""Approved training-start controllers and an EXPERIMENTAL screening box.

Candidate 2 was accepted as the Actor/Critic learning starting point
after fixed-NMPC tests on TRAIN. The good fixed NMPC comparator remains
completely unchanged. This module NEVER overwrites baseline.py defaults.

A screening box is only a proposed candidate domain, NOT proof of paper
Assumptions 3-4, an invariant terminal set, recursive feasibility, or
stability for every weight within the box.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

TRAIN_REFERENCE_SHA256 = (
    "afa67ac24ef61b591b1a16498bcb1ee5e521452cf0f8036de00923d062940b98"
)


@dataclass(frozen=True)
class FixedWeights:
    q: tuple
    r: tuple
    p: tuple

    def arrays(self):
        return tuple(np.array(x, dtype=float) for x in (self.q, self.r, self.p))


# Actual config.json uploaded by user for the second candidate.
LEARNING_INITIAL = FixedWeights(
    q=(12., 3., 7.5, 6., 100., 100.),
    r=(.6, 32.),
    p=(96., 24., 60., 48., 800., 800.),
)

# Independent strong baseline from unchanged fixed NMPC in baseline.py.
GOOD_NMPC = FixedWeights(
    q=(12., 3., 15., 6., 183.75, 192.5),
    r=(.6, 16.),
    p=(180., 45., 225., 90., 2756.25, 2887.5),
)


def _validate(w):
    arrays = w.arrays()
    for name, value, shape in zip(("Q", "R", "P"), arrays, (6, 2, 6)):
        if value.shape != (shape,) or np.any(value <= 0.) or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid {name} diagonal")
    return arrays


def screening_box(margin_fraction: float):
    """Return positive candidate lo/hi for Q[6],R[2],P[6].

    The smallest coordinatewise box containing the learning initial
    controller AND good fixed comparator is padded symmetrically by
    margin_fraction of each endpoint. This is *only* an offline
    screening DESIGN, never the automatically approved RL action range.

    E.g., margin_fraction=0.20 yields [0.8*min,1.2*max] for each
    coordinate, without asserting every point is feasible.
    """
    margin = float(margin_fraction)
    if not np.isfinite(margin) or not (0. < margin < 1.):
        raise ValueError("Require explicitly supplied 0 < margin_fraction < 1")
    start = _validate(LEARNING_INITIAL)
    best = _validate(GOOD_NMPC)
    lo = tuple((1. - margin) * np.minimum(a, b) for a, b in zip(start, best))
    hi = tuple((1. + margin) * np.maximum(a, b) for a, b in zip(start, best))
    for a, b, low, high in zip(start, best, lo, hi):
        if np.any(low <= 0.) or np.any(a <= low) or np.any(a >= high):
            raise AssertionError("Initial weights not interior")
        if np.any(b <= low) or np.any(b >= high):
            raise AssertionError("Baseline not interior")
    return lo, hi


def representative_weights(margin_fraction: float, profile="full"):
    """Named finite screening SAMPLE, not proof for its convex hull.

    Quick: two endpoints and midpoint, at the previously tested initial
    condition. Full: 9 carefully chosen combinations including corners;
    each will be simulated under several initial vehicle offsets.
    """
    lo, hi = screening_box(margin_fraction)
    a = LEARNING_INITIAL.arrays()
    b = GOOD_NMPC.arrays()
    midpoint = tuple((x + y) / 2. for x, y in zip(a, b))
    results = {
        "learning_initial": a,
        "strong_fixed_baseline": b,
        "coordinate_midpoint": midpoint,
    }
    if profile == "full":
        results.update({
            "all_lower_corner": lo,
            "all_upper_corner": hi,
            "stage_lower_terminal_upper": (lo[0], lo[1], hi[2]),
            "stage_upper_terminal_lower": (hi[0], hi[1], lo[2]),
            "start_stage_good_terminal": (a[0], a[1], b[2]),
            "good_stage_start_terminal": (b[0], b[1], a[2]),
        })
    elif profile != "quick":
        raise ValueError("profile must be quick or full")
    return results
