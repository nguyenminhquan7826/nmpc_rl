# TRAIN-only discounted-return diagnosis of the Direct-P Critic

**Status:** mathematical diagnostic implemented, unit tests and CI pass.
The actual 27 full-screening closed-loop CSVs are local on the user's
Windows checkout, not in this GitHub branch. No one has run the complete
return audit on the user's 27 local trajectories in GitHub CI.

This does NOT modify the baseline NMPC, Actor network, Critic TD(0)
learning rule, reward, gamma, or learning rates. It does NOT train a
new Critic. No held-out TEST data is used.

## 1. Rationale

The user selected a six-scalar, DIRECT (not NN) quadratic Critic:

```
V_P(s) = -sum_i P_i * e_i(s)^2
Qf = diag(P), P > 0.
```

The paper reward uses fixed, independent evaluation coefficients:

```
r_k = -(e_k^T Qe e_k + du_k^T Re du_k)
Qe = [20,20,100,20,600,600]   # legacy experiment CANDIDATE
Re = [50,100]                 # legacy experiment CANDIDATE
```

Ten-step stochastic training smoke on pinned TRAIN had:
`mean_reward=-1.4972644`, `rms_td=1.4711452`, solver success
100%, maximum single-step `||Delta P||=3.83e-6` at
`critic_lr=.001`. A tiny P update is not itself evidence of a
bug; before changing learning rates, test whether *ANY* six-P
quadratic within the screened box can explain cost-to-go.

## 2. Formula and three fits

For each full fixed-NMPC TRAIN trajectory of recorded rewards
`r[0], ..., r[T-1]`, compute the positive observed finite-episode
discounted cost-to-go:

```text
C[T] = 0
C[t] = -r[t] + gamma * C[t+1], for t=T-1,...,0.
```

Compare `C[t]` against `sum_i P_i * e_i[t]^2`:

1. **Initial fixed P** from the user-approved candidate 2:
   `[96,24,60,48,800,800]`. No parameter fitting.
2. **Box-constrained best fitting P** from the 27 TRAIN trajectories.
   Coordinate descent minimizes retrospective squared approximation
   error subject to the formerly screened per-coordinate lower/upper
   bounds. This is *offline diagnostic fitting*, NOT training TD(0),
   NOT the next Critic initialization or an approved replacement P.
3. **Signed, unbounded least-squares P** as a mathematical
   unconstrained lower bound on squared error. It may contain
   negative/large coefficients and MUST NOT be used as an NMPC
   terminal cost. The linear regression rank is also reported.

Report RMSE, MAE, normalized RMSE relative to RMS of the target,
R-squared where defined, and coordinates pinned at the lower/upper
box. An unconstrained regression fit may still have substantial
residuals: that indicates the six-state diagonal quadratic family
cannot reproduce the dataset perfectly even without an admissible
box (subject to finite recorded-return caveat).

The tool also conducts **leave-one-initial-condition-out within
TRAIN**: for each initial condition, fit the box on the other two
conditions across all nine fixed-weight policies, then evaluate on the
held-out TRAIN condition. This is a diagnostic for dependence on
initial condition; it is NOT the final held-out TEST evaluation.

### Important interpretation limits

- Fixed-policy trajectories do NOT establish performance, TD target,
  or value-function errors under a stochastic adaptive Actor.
  This is a prerequisite scale/capacity check only.
- The fixed-NMPC log records pre-step state errors and the input
  increment actually applied that step. It does not store the
  exact last post-step state. With `reached_end=true`, this
  diagnostic includes the last logged reward and assumes no
  unobserved future reward. It is a finite-episode return, not a
  proof of the infinite-horizon value/terminal return. Boundary
  effects are most pronounced at the last steps.
- `gamma` changes the SCALE of discounted returns. **Do not choose
  gamma merely by the smallest raw RMSE** across gamma sweeps.
  Compare dimensionless NRMSE, constraints, TD scales, the
  sampling period `Ts=0.071s`, and the task interpretation.
- The six-P Critic does not explicitly observe previous
  `u` (although the Actor's 17-feature observation does). The
  step reward explicitly penalizes `du`. Even unbounded P may not
  encode all return variation attributable to changing controls or
  future path curvature.
- A box screened at 27 *fixed* controllers does NOT prove
  stability/invariance when Actor and Critic change weights every
  sampling time.
- None of the fit outputs authorizes automatically changing reward
  Qe/Re, discount gamma, Actor NN sigma, or learning rates.

## 3. Exact PowerShell command

From the project's `nmpc_casadi` working directory:

```powershell
git status --short
git pull --ff-only

python -m unittest -v test_critic_return_audit

python audit_critic_returns.py `
  --screen-root .\results_screen\full_20pct `
  --output .\results_critic_audit\gamma_sweep_v1 `
  --gammas 0.90 0.95 0.99
```

The script requires the original local TRAIN CSV (the exact
SHA-256 `afa67ac24ef61b591b1a16498bcb1ee5e521452cf0f8036de00923d062940b98`),
27 `EMPIRICAL_PASS` full-screening rows, and each run's unchanged
`config.json`, `summary.json` and `nmpc_fixed.csv`.

Outputs in the new (empty) folder:

- `summary.json`: per-gamma 6P fits, absolute and normalized
  errors, number of saturated bounds, fit convergence, and
  leave-one-initial-condition-out results.
- `return_gamma_comparison.csv`: compact side-by-side summary
  for specified gammas.
- `return_cases.csv`: per-gamma per-case starting and fitted
  value predictions, first and final observed returns, RMSE.

## 4. Decision after results

Review together:
- If constrained fit remains inaccurate AND the signed
  unbounded fit is substantially better, the **screening box**
  likely limits approximation on TRAIN. Do NOT expand it without
  renewed feasibility and terminal-set analysis.
- If constrained and unrestricted fits both have large normalized
  residual, it suggests a **representation / feature or
  policy-conditioning limitation**, not merely a learning-rate
  problem.
- If constrained fit is adequate but P barely changes in
  10-step live training, then analyze TD gradient magnitudes,
  learning rates and run a longer monitored TRAIN-only pilot.
- If all three fit error scales vary by gamma, interpret
  side-by-side with task horizon (at `Ts=.071s`) and
  finite-horizon boundary effects. Do not pick a gamma from a
  raw reward magnitude alone.

After reviewing the emitted files, confirm numeric hyperparameters
with the user before any 100–200-step learning pilot.
