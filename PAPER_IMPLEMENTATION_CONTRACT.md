# Source-to-implementation contract — Actor–Critic–NMPC rebuild

**Status:** Source-mapped mathematics plus implemented six-parameter direct Critic and NMPC weight bridge; no Actor NN or training yet.
**Sources of authority:** User's `MPC_RL.pdf` (9 pages), and
`NMPC(1).docx` (37 pages). The paper has higher priority where the two
disagree. **No unilateral architecture or numerical change.**

**Immutable reference baseline:** `experiment/direct-quadratic-td-critic`
at `10831307b313e4355b912e21c6016b130c590f88` (NMPC-only,
existing source plus smoke tests). Keep this branch clean so results remain
comparable and reproducible.

## Confirmed directly from the paper

| Paper reference | Implementation contract | State |
|---|---|---|
| Eq. (1)-(2), (20)-(28) | Receding-horizon NMPC, nonlinear plant, six-state tracking, quadratic stage/control increment and terminal costs | Existing NMPC; compare indexing later |
| Eq. (3) | RL's MDP state is separate from the physical state; actor's action is a cost-weight parameter | Requires precise observation definition |
| Eq. (4) | `r=-(e.T Qe e + du.T Re du)`; use fixed task-evaluation matrices `Qe,Re`, not an auto-varying task reward | `rl_paper_core.paper_reward` |
| Eq. (5) | Maximize expected discounted task return | Architecture-agnostic |
| Eq. (6) | `delta=r+gamma*V(s_next)-V(s)`; TD(0) | `rl_paper_core.td_error` |
| Eq. (7) | `L_critic=0.5*delta²` | `rl_paper_core.critic_squared_td_loss` |
| Eq. (8) | Policy-gradient ascent proportional to `delta*grad_actor(log pi_actor)` | Scalar loss identity tested; actor not built |
| Figure 3; Section V(a); Algorithm 1, steps 4-7 | Actor tunes `Q,R`; Critic tunes `Qf`; NMPC chooses `u=[a,delta]` | Six directly learned Critic P parameters and parameterized NMPC bridge implemented; Actor pending |
| Eq. (29)-(33) | Positive weight matrices and Euclidean projection onto an offline-screened compact admissible parameter set | Box projection implemented, but NOT certified stable |
| Assumptions 2-4, Eq. (39)-(40) | Terminal invariant set/local control law/terminal Lyapunov decrease required for cited stability theorem | **NOT implemented or verified** in legacy NMPC. Do not claim theorem holds |

### Sourced numerical facts, not implicit adaptations

- Paper Section VI MATLAB experiment: `Ts = 0.01 s`, `N = 10`,
  vehicle `m = 1500 kg`, `Iz = 3000`, `Cf = 38800`,
  `Cr = 66000`; this repository simulates an approximately 1:10
  platform with `Ts = 0.071 s`, `N = 10`. These are NOT numerically
  identical experiments; maintain transparent distinction.
- Paper specifies **stochastic** actor policy. The user's document,
  Section II, pages 34-36, specializes it to a Gaussian actor with
  NN-produced mean **and learned standard deviation**.
- Paper **Remark 4 explicitly allows any parameterized critic architecture**.
  The user's document, pages 36-37, specializes this to an NN that outputs
  terminal weights. Prior project design used six directly trainable P
  parameters instead, which may be an allowed parameterization but is not
  the NN architecture described by the document.
- The paper does not sufficiently prescribe NN layer counts, minibatch
  horizon, Adam/SGD hyperparameters, activation function, exact reward
  weights, or numerical admissible bounds for the user's scaled vehicle.
  These must be identified as implementation choices and **approved first**.

### Paper notation inconsistencies requiring explicit interpretation

- Algorithm 1, step 13 prints a gradient indexed by the **Critic**
  parameters even though it acts on the **Actor** log policy. The
  mathematical policy-gradient relation in Eq. (8) implies differentiation
  with respect to Actor parameters. This appears to be a subscript error;
  document interpretation explicitly before implementing gradients.
- Eq. (14) typesets the world-frame Y velocity with a minus sign whereas
  Eq. (19) and kinematic rotation use a plus sign.
- The rear lateral force formula and axle-length symbol in Eq. (17)/(19)
  are not fully self-consistent. The baseline deliberately preserves the
  pre-existing `restoring` sign convention until verified.
- The paper's stability claims are conditional on invariant terminal
  ingredients. Weight box clipping alone is **not** a stability proof.

## Staged, test-driven implementation gate

1. **Done:** preserve NMPC-only branch and source-register differences.
2. **Done:** implement architecture-independent paper Eq. (4), (6)-(8),
   projection Eq. (29)-(33), deterministic unit tests.
3. **DONE / user-approved:** Critic has **exactly six directly trained**
   terminal coefficients `p_i` and NO NN: `P=diag(p1,...,p6)`,
   `V_P(s)=-e(s)^T P e(s)`, and **the SAME** `Qf=P` passes to NMPC.
   TD(0) semi-gradient: `dL/dp_i=delta * e_i^2` with bootstrap target
   frozen, then `p <- project_box(p - critic_lr * gradient)`.
   Bounds, gamma and critic learning rate are caller-supplied; **no**
   undocumented default. Implemented in `nmpc_casadi/direct_p_critic.py`
   and connected through `critic_nmpc_bridge.py`, using optional adaptive
   diagonals in `nmpc_casadi.py`; fixed default remains unchanged.
4. **Still requires decisions grounded in source:** actor observation `s`,
   action dimensions, Gaussian mean/scale map, admissible regions,
   training schedule, terminal handling, and vehicle scaling.
5. Only then build Actor/Critic and connect a separate RL driver to NMPC;
   use identical fixed baseline/track/plant for before/after comparison.
6. Full held-out validation without reusing test data to select models.

**No change to the validated NMPC solver or vehicle dynamics is authorized
by implementing this module.**

## Explicit deviation from user's NMPC(1).docx pages 36–37

| Issue | Document's pages 36–37 | Approved direct-six-parameter Critic |
|---|---|---|
| Trainable architecture | Neural Network producing six terminal weights from MDP state | Six physical positive scalar parameters **only**; no network layers, biases, learned baseline, or state-conditioned output |
| Terminal weights | `Qf_theta(s)` depends on current observed state through NN | `Qf=P=diag(p1,...,p6)` is global while trained; changes after TD update, not directly as a function of `s` |
| Value | `V_theta(s)=-e(s)^T Qf_theta(s)e(s)` | `V_P(s)=-e(s)^T P e(s)` |
| Critic optimization | Backpropagation through NN output and value | Direct semi-gradient from squared one-step TD loss, `grad_p_i = delta*e_i^2`, projected onto supplied positive box |
| State feedback | NN changes terminal weights for different states | State affects value through `e(s)` and influences later P updates, **but identical P for different states before an update** |
| Paper authority | Specific implementation from explanatory document | Remark 4 in MPC_RL.pdf expressly permits *any parameterized critic architecture* |

This is the user's **explicit 2026-10-20 architecture decision**, not an
inference from paper Eq. (6). It is a permitted specialization under Remark
4 rather than an exact reproduction of document pages 36–37.

**Numerical and methodological limitations:**
- The 6 diagonal P entries are **physical Qf weights** with state order
  `[vx,vy,psi,r,X,Y]`, not a hidden normalized/log/multiplier space.
- For reference error `e`, wrap heading to `(-pi,pi]` as in existing NMPC.
- Given reward `r`, `delta=r+gamma V_P(next)-V_P(current)`.
  In a genuinely terminal transition, substitute `V(next)=0`; retain
  bootstrap at a time-limit truncation. Paper Eq. (6) omits episode endings.
- Squared TD loss is `0.5*delta**2`; **do not** differentiate the bootstrap
  target (`delta*e_i**2` is *semi-gradient*, not full residual gradient).
- An externally supplied projection box is necessary but does **not**
  imply Assumptions 3–4 / terminal invariant set are satisfied. Existing
  fixed NMPC cannot yet be called stability-certified.
- Future training should test whether constant P has enough expressive
  power for this path tracking problem. It may be inadequate even if
  its TD optimization is implemented correctly. This is a scientific
  limitation, not evidence that the implementation violates Remark 4.
- Paper's MPC formulas include factors of 1/2 in Algorithm 1, while the
  preexisting CasADi NMPC cost uses no such *global* factor. The bridge
  DOES NOT rescale any coefficients or alter the historical baseline.
  The bridge only passes six learned Qf entries into the SAME objective.

## Approved Actor action space — 8 independently adapted diagonal weights

**Decision (explicit user approval, 2026-10-21):** The Actor independently
adapts ALL six diagonal stage-state penalties and BOTH input-increment
penalties, in order:

- `Q = diag(q_vx, q_vy, q_psi, q_r, q_X, q_Y)`
- `R = diag(r_delta_a, r_delta_steer)`

The Actor action/parameter dimension is **8**. Critic is unchanged:
`Qf = diag(p_vx, p_vy, p_psi, p_r, p_X, p_Y)`, exactly 6 direct
TD(0)-learned parameters. Paper Section V and Algorithm 1 require
Actor adaptation of Q/R, but **do not specify 8 independent scalars**;
this is an explicit project design choice.

## Actor observations and MLP — proposal awaiting user approval

Paper Eq. (3), Section II(b): the RL MDP state includes both system states
and tracking errors. Remark 7 says the MDP state is distinct from the
physical system state. Paper does not fix the observation vector size
or NN layer widths. The user document pages 34–36 specifies an Actor
Gaussian policy whose neural network learns BOTH mean and dispersion.

Proposed **17-dimensional** observation, pending sign-off:

`s_RL = [x(6), e(6), u_previous(2), path_progress(1),
          curvature_now(1), curvature_at_horizon_end(1)]`.

- `x=[vx,vy,psi,r,X,Y]`; `e=x-xref`, with heading wrapped.
- Previous `[a,delta]` is relevant to input-difference cost/constraints.
- Arc-length path progress and preview curvature are added as
  implementation choices to distinguish upcoming reference geometry;
  the paper **does not specify them**.
- Whether this finite vector is sufficiently Markov is not proven.
  Speed-varying/longer preview may demand additional reference information.
- For simulation, x is presently the model state; a real observer
  and sensor estimates remain a separate implementation task.

Proposed Actor Gaussian NN — **not yet approved or coded**:
`17 input -> two fully connected hidden layers of width 64 with tanh
-> mu(8) and dispersion_raw(8) output heads` (16 numbers).
Dispersion is constrained positive in the Gaussian policy.
Sample latent vector `z = mu + sigma * standard_normal(8)`;
map z into the eight strictly positive admissible weight intervals,
e.g. smooth squashing, with learned mu/sigma and correct policy
log-probability for the sampled latent vector.

The **two-by-64 hidden architecture**, activation function, Gaussian
scale limits, squash details, numerical weight bounds, learning rates,
and policy initialization are **implementation proposals**, NOT
equations prescribed by the paper. Do not write Actor training code
or claim its admissible set is stability-certified before approval
and the paper's offline feasibility/terminal-conditions screening.
