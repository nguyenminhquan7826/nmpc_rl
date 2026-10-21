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
RESULTS_DIR = BASE_DIR / "results_v6"
IMAGES_DIR = BASE_DIR / "images_v6"
TRAIN_CSV = TRAJECTORY_DIR / "test_xref.csv"
# Optional only if your original pinned TRAIN lives in another checkout
# and the bounded automatic local search cannot locate it:
# TRAIN_CSV_SOURCE = Path(r"D:\\your\\old\\project\\reference_split\\train_xref.csv")
TRAIN_CSV_SOURCE = None

# Keep strictly away from TEST until all training decisions are fixed.
EPISODES = 100
STEPS_PER_EPISODE = 1150
CHECKPOINT_EPISODES = (1, 10, 25, 50, 100)
GAMMA = 0.90
SEED = 0
INITIAL_EY_M = .05
INITIAL_HEADING_DEG = 0.

# Experiment D: v3-matched 100 episodes, margin and optimizer rates.
# The two TRAIN-only penalties are exploratory; no stability certificate.
ACTOR_LR = 1e-5
CRITIC_LR = 1

# Per-component learning-rate multipliers (controlled ablation).
# Actor order: Q[vx,vy,psi,r,X,Y], R[acceleration,steering].
# The eight multipliers precondition BOTH Gaussian mean/sigma gradients
# and their contributions to the shared hidden layers.
# CRITIC order: Qf=P[vx,vy,psi,r,X,Y]; direct six-parameter TD updates.
#
# Keep per-component factors identical to the completed v3 run.
# Outputs must be new empty directories; historical v3 results stay intact.
ACTOR_OUTPUT_LR_FACTORS = (1., 1., 0.5, 0.5, 1, 1, 0.5, 0.5)
CRITIC_P_LR_FACTORS = (0.5, 0.5, 0.1, 0.2, 1, 1)
SIGMA_MIN = .02
SIGMA_INIT = .10
SIGMA_MAX = .30

# Previously screened 20%-padded DESIGN, not all-box proof of stability.
WEIGHT_BOX_MARGIN = 0.40
REWARD_CANDIDATE = "legacy"  # constant Qe/Re in audit_training_reward.py
EVALUATE_SAME_TRAIN_TRACK = True

# Experiment D ONLY in TRAIN: frozen rollout still uses fixed paper Eq.(4).
SOLVER_FAILURE_PENALTY = 10.0      # only after both solver attempts fail
FAILURE_TERMINAL_PENALTY = 100.0  # additional at catastrophic termination
ACTOR_MU_REG_COEF = 1e-3          # square hinge in latent mean mu
ACTOR_MU_REG_THRESHOLD = 2.0      # penalty begins when abs(mu)>2
