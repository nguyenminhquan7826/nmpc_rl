# Source-to-implementation contract — Actor–Critic–NMPC rebuild

**Status:** Mathematical core only; no Actor or Critic network, no training.
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
| Figure 3; Section V(a); Algorithm 1, steps 4-7 | Actor tunes `Q,R`; Critic tunes `Qf`; NMPC chooses `u=[a,delta]` | Pending architecture |
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
3. **Requires decision:** Critic = NN outputting six positive `Qf(s)`
   entries (document Section II) **or** six directly optimized coefficients
   `P` (prior project architecture).
4. **Requires decisions grounded in source:** actor observation `s`,
   action dimensions, Gaussian mean/scale map, admissible regions,
   training schedule, terminal handling, and vehicle scaling.
5. Only then build Actor/Critic and connect a separate RL driver to NMPC;
   use identical fixed baseline/track/plant for before/after comparison.
6. Full held-out validation without reusing test data to select models.

**No change to the validated NMPC solver or vehicle dynamics is authorized
by implementing this module.**
