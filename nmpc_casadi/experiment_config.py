"""One place to edit the experiment; no per-run shell path arguments.

User-facing folders beside this Python code:
  reference_split/   original pinned TRAIN trajectory CSV (never overwritten)
  results/           numerical results, selected checkpoints and CSV logs
  images/            tracking and Q/R/Qf plots

Old exploratory results_screen / results_rl folders are historical and
are NOT read by this simplified runner. The 20%-margin weight box was
designed from prior 27/27 sampled TRAIN fixed-NMPC screening; embedding
its design does NOT convert screening into a stability certificate.
This configuration does not silently tune reward or hyperparameters.
"""
from __future__ import annotations
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TRAJECTORY_DIR = BASE_DIR / "reference_split"
RESULTS_DIR = BASE_DIR / "results"
IMAGES_DIR = BASE_DIR / "images"
TRAIN_CSV = TRAJECTORY_DIR / "train_xref.csv"
# Optional only if your original pinned TRAIN lives in another checkout
# and the bounded automatic local search cannot locate it:
# TRAIN_CSV_SOURCE = Path(r"D:\\your\\old\\project\\reference_split\\train_xref.csv")
TRAIN_CSV_SOURCE = None

# Keep strictly away from TEST until all training decisions are fixed.
EPISODES = 10
STEPS_PER_EPISODE = 1150
CHECKPOINT_EPISODES = (1, 5, 10)
GAMMA = 0.90
SEED = 0
INITIAL_EY_M = .05
INITIAL_HEADING_DEG = 0.

# Gamma .90 pilot: change Critic Qf update rates ONLY. Actor is frozen\n# in learning-rate *settings*, NOT frozen NN weights; it keeps training.
ACTOR_LR = 1e-6
CRITIC_LR = 1e-2

# Per-component learning-rate multipliers (controlled ablation).
# Actor order: Q[vx,vy,psi,r,X,Y], R[acceleration,steering].
# The eight multipliers precondition BOTH Gaussian mean/sigma gradients
# and their contributions to the shared hidden layers.
# CRITIC order: Qf=P[vx,vy,psi,r,X,Y]; direct six-parameter TD updates.
#
# Nonuniform Critic rates below are an exploratory 10-episode ablation.
# They are NOT certified safe/optimal; compare with the previous 100-ep
# baseline and check NMPC solver success, return, relative Qf changes.
# Keep Actor factors as one to isolate the Critic learning-rate change.
# Results/images must be empty, preserving your previous experiment.
ACTOR_OUTPUT_LR_FACTORS = (1., 1., 1., 1., 1., 1., 1., 1.)
CRITIC_P_LR_FACTORS = (1.23, 0.86, 0.25, 0.41, 1.84, 1.85)
SIGMA_MIN = .02
SIGMA_INIT = .10
SIGMA_MAX = .30

# Previously screened 20%-padded DESIGN, not all-box proof of stability.
WEIGHT_BOX_MARGIN = .20
REWARD_CANDIDATE = "legacy"  # constant Qe/Re in audit_training_reward.py
EVALUATE_SAME_TRAIN_TRACK = True
