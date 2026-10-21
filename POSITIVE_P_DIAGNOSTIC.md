# TRAIN-only follow-up: positive six-P and policy dependence

**Input evidence:** the user ran `audit_critic_returns.py` over the
27 previously screened fixed-NMPC TRAIN trajectories (27,867 logged
pre-step states) and uploaded the three outputs in the ChatGPT
conversation on 2026-10-21.

| Gamma | Initial P RMSE | Best box-P RMSE | Signed unrestricted RMSE | Box normalized RMSE | Box R2 |
|---|---:|---:|---:|---:|---:|
| 0.90 | 7.3583 | 4.6814 | 0.4983 | 0.5707 | 0.2693 |
| 0.95 | 14.8644 | 12.2553 | 1.6016 | 0.7813 | -0.4563 |
| 0.99 | 61.7369 | 59.5706 | 26.7697 | 0.9544 | -3.1508 |

At **all three** gamma values the box-constrained retrospective P fit
hit ALL SIX box upper bounds:

```text
P_box = [216, 54, 270, 108, 3307.5, 3465]
```

At gamma=.99 the recorded average finite-episode cost-to-go is about
55.146 but the initial quadratic P predicts 0.636 and the saturated
box-P predicts only 2.683. The old live 10-step Critic TD update was
tiny, but increasing its learning rate cannot by itself remove the
representational discrepancy of the fixed box.

Signed unrestricted regressions include negative P entries:
- gamma=.90: `p_r=-532.0`
- gamma=.95: `p_vx=-383.4,p_r=-1646.4`
- gamma=.99: `p_vx=-44417.7,p_psi=-277.8,p_r=-8631.1`

**None of these signed P fits may be used as terminal weight matrices.**
The unrestricted `R2=.1618` at gamma=.99 additionally suggests that
even a signed linear quadratic P fit on a **pool of nine different
fixed controller policies** is insufficient on these finite-episode
returns. This is not a proof of Critic incapability on a single
adaptive Actor policy.

The number of future control cycles weighted by gamma increases
strongly across .90,.95,.99 (at Ts=.071s, rough geometric horizons
Ts/(1-gamma) = .71s,1.42s,7.1s). Gamma must be chosen for
the research objective rather than comparing raw RMSE across
differently scaled return targets.

## New follow-up diagnostic: physically nonnegative, no upper limit

We need to disentangle THREE possibilities before changing
learning rates, discount gamma, or terminal P bounds:

1. **Original P box:** all six upper limits are saturated.
2. **Positivity:** unconstrained fits partly exploit *negative* P,
   which the paper's terminal cost cannot use.
3. **Policy mixing:** the pooled dataset consists of nine distinct
   *fixed* NMPC policies. Their values at a common state need not
   agree; a single six-error Critic cannot condition on the
   policy choice. Thus pooled residuals also reflect policy mixing.

`nmpc_casadi/audit_positive_critic_fit.py` now computes the
best six-dimensional least-squares fit with `P>=0` and no upper
bounds using 64-support enumerated QR (no SciPy dependency).
This is a **mathematical diagnostic**, not a feasible/stable
NMPC controller. It also fits a separate such model for each
of the nine fixed policies so pooled- and within-policy
residuals can be compared. A KKT check is included for the
nonnegative constrained optimum.

From the current `nmpc_casadi` directory on the Windows machine:

```powershell
git pull --ff-only

python -m unittest -v test_positive_critic_fit test_critic_return_audit

python audit_positive_critic_fit.py `
  --screen-root .\results_screen\full_20pct `
  --output .\results_critic_audit\positive_p_v1 `
  --gammas 0.90 0.95 0.99
```

Outputs:
- `summary.json`: six positive-no-upper fit coefficients, pooled
  diagnostics, per-policy fits and checks; **all marked diagnostic only**.
- `positive_gamma_comparison.csv`: quick pooled fit comparison
  across gamma.
- `positive_policy_comparison.csv`: which fixed policies fit worse
  with one pooled P than with policy-specific P.

Interpretation AFTER the user supplies the results:
- If positive-unbounded P substantially improves on box-P, then
  upper P limits are an important statistical bottleneck, but
  cannot simply be widened without renewed terminal admissibility
  and feasibility analysis.
- If positive-unbounded P is much worse than signed-unbounded P,
  positivity is important. Dropping it violates the user-chosen
  terminal quadratic and paper assumptions.
- If each fixed-policy fit is good but pooled fit is poor, policy
  dependence is a major explanation. The current Direct-P
  architecture has no policy-conditioned inputs.
- If within-policy positive-unbounded fits also remain poor, the
  value-function form may not represent the finite-return dependence
  of this tracking task, even with arbitrary positive magnitude.

**Do not promote positive-unbounded fitted P into the RL box,
do not retrain or modify reward/learning rates automatically,
and never use TEST to choose any of these settings.**
