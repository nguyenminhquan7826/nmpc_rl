# Actor–Critic learning start and Q/R/P offline screening

**Status:** the user approved candidate 2 as the official start for the
FIRST learning experiment, and the original strong NMPC remains a separate
immutable comparator. The actual joint training driver is NOT run yet.
Do not mislabel offline finite screening as a proof of closed-loop stability.

## Confirmed project decisions and provenance

The user ran fixed-weight NMPC with `reference_split/train_xref.csv`
(`Ts=0.071 s, N=10`, `ey=0.05 m`, `heading=0 deg`) and uploaded its
`config.json` and summary. That configuration established this **approved
learning initialization** (all physical coefficients, state order
`[vx,vy,psi,r,X,Y]` and control-increment order `[a,steer]`):

| Matrix | Diagonal |
|---|---|
| `Q_start` | `[12,3,7.5,6,100,100]` |
| `R_start` | `[0.6,32]` |
| `P_start=Qf_start` | `[96,24,60,48,800,800]` |

The unchanged **strong fixed NMPC comparator** is:

| Matrix | Diagonal |
|---|---|
| `Q_good` | `[12,3,15,6,183.75,192.5]` |
| `R_good` | `[0.6,16]` |
| `P_good=15*Q_good` | `[180,45,225,90,2756.25,2887.5]` |

These matrices are pinned in `nmpc_casadi/training_initialization.py`.
They do not change `baseline.py`, the NMPC-only CLI defaults, existing
constraints, or the separate comparator branch.

In the uploaded TRAIN runs:

| Measure | Candidate 2 | Strong fixed |
|---|---:|---:|
| `mean_cte_post_m` | 0.0200309671 | 0.0195463382 |
| `max_cte_post_m` | 0.0377498762 | 0.0330598939 |
| `rmse_heading_post_rad` | 0.0602033389 | 0.0522585531 |
| `rmse_vx_post_mps` | 0.0079173562 | 0.0134343690 |
| `solver_success_pct` | 100 | 100 |
| `tracking_feasible` | true | true |

Candidate 2 is worse at some tracking measures, but better at speed
tracking. There is no basis to assume a reward improvement or inferior
whole-objective performance without a fixed common task reward.

TRAIN is **locked by checksum**:
`afa67ac24ef61b591b1a16498bcb1ee5e521452cf0f8036de00923d062940b98`.
`test_xref.csv` must NEVER be used for weight-bound selection, screening,
or checkpoint selection. The TRAIN CSV is local to the user's Windows
checkout and is **not** stored in this branch, so GitHub CI validates only
the screening code and independent mathematical tests, not 27 actual
TRAIN trajectories.

## Proposed screening box — NOT yet an approved learning box

A numerical candidate family is constructed only for offline testing as

`w_lower = 0.8 * min(w_start,w_good)`,
`w_upper = 1.2 * max(w_start,w_good)`.

This makes both sets of weights strictly interior, including dimensions
where the two sets initially coincide.

| Group | Proposed LOWER | Proposed UPPER |
|---|---|---|
| Q six | `[9.6,2.4,6,4.8,80,80]` | `[14.4,3.6,18,7.2,220.5,231]` |
| R two | `[0.48,12.8]` | `[0.72,38.4]` |
| P six | `[76.8,19.2,48,38.4,640,640]` | `[216,54,270,108,3307.5,3465]` |

The **20% margin is a screening hypothesis chosen for exploration**, not
a paper number, not the user's approval of final bounds, and not
evidence of stability. In particular, testing the two endpoint
controllers is NOT enough to validate cross-combinations or every point
in this 14-dimensional box.

The `quick` profile runs 3 weight configurations (candidate 2, strong
baseline, midpoint) at one previously used initial condition
(`ey=0.05 m`, `heading=0 deg`). The `full` profile runs 9
configurations × 3 initial offsets = **27 closed-loop trajectories**;
the additional points include lower/upper weight corners, opposite
stage/terminal extremes, and cross-combinations of start vs good
terminal weights. The conditions include `ey=+/-0.05 m` and
`heading=5 deg`. All use the fixed NMPC solver, not RL.

A sample passes this **empirical** filter only if the full run
reaches the end, the solver succeeds at >=99.9% of steps, does not
abort, and existing `tracking_feasible` criteria are satisfied.
These screening thresholds are experiment-level decisions and may
later require revision. Passing alone does NOT validate the entire
parameter box.

## Running on the user's Windows computer

Because the user is already in
`D:\GitHub\RL_NMPC\Code\direct-quadratic-td-critic\nmpc_casadi`
and already switched to the experiment branch:

```powershell
git status --short
git pull --ff-only
python -m unittest -v test_screen_training_weights test_gaussian_weight_actor test_direct_p_critic test_rl_paper_core test_nmpc_core

# First confirm the pinned TRAIN file and run three cases:
python screen_training_weights.py `
  --reference .\reference_split\train_xref.csv `
  --output results_screen\quick_20pct `
  --margin 0.20 --profile quick

# Run only after checking the 3-case quick screening:
python screen_training_weights.py `
  --reference .\reference_split\train_xref.csv `
  --output results_screen\full_20pct `
  --margin 0.20 --profile full
```

Outputs: `manifest.json` with the exact reference checksum, numerical
screening envelope, individual sampled scenarios and explicit
non-certification note; `screening.csv` with each sampled run and
pass/fail; `<case>/config.json`, `<case>/summary.json`, and
`<case>/nmpc_fixed.csv` for replay/diagnostics.

Use fresh output directories. The script refuses a different reference
SHA256 (including a changed or held-out CSV). It reports failures, does
not silently discard failed cases, and exits nonzero if any case fails.

## Conditions for proceeding to joint Actor–Critic training

1. Inspect TRAIN screening and revise the exploratory box if
   necessary. Do not advertise 27 finite simulations as the terminal
   invariant set, local stabilizing controller or recursive feasibility
   proof required by the paper.
2. Separately agree/adopt numerical `Q/R/P` bounds for RL based on
   screening and the paper's terminal assumptions. The direct-P Critic
   requires six physical P bounds; Actor requires six Q + two R bounds.
3. Choose/approve positive Gaussian `sigma_lower, sigma_initial,
   sigma_upper`; the Actor constructor purposefully has no default.
4. Fix paper reward evaluation weights `Qe,Re`, discount factor,
   Actor NN optimizer/learning rate, Critic learning rate, and sampling
   and episode protocol. Train on TRAIN, select models on a proper
   validation set, evaluate on held-out TEST only after these choices
   are frozen.
5. Only then build and run the joint training loop; report improvement
   against both the fixed candidate 2 and the unchanged strong NMPC
   comparator using identical dynamics, reference, initialization and
   reward definitions. Prefer the paper's TD(0) and score-function
   objective without unapproved regularizers or reward shaping.
