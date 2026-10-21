"""One verified TD(0)+Gaussian policy-gradient transition, no reward shaping.

Actor uses the *same pre-update* TD as Critic. The sampled Gaussian latent
action is detached in the score function. The 6 physical Critic coefficients
are updated by projected semi-gradient; all 6352 Actor NN coefficients by
explicit-step SGD using the source Eq. (8). No gradient through NMPC.

Optimizer (SGD) and learning-rate magnitudes are user-experiment inputs;
this module does NOT claim that paper prescribes numerical values.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from gaussian_weight_actor import GaussianWeightActor
from direct_p_critic import SixParameterTDCritic
from rl_paper_core import td_error


@dataclass(frozen=True)
class TrainingStep:
    td: float
    actor_loss: float
    critic_loss: float
    actor_grad_norm: float
    actor_param_step_norm: float
    critic_param_step_norm: float
    value_current: float
    value_next: float
    reward: float


def learn_one_transition(
    actor: GaussianWeightActor,
    critic: SixParameterTDCritic,
    observation, sampled_latent,
    error, next_error, reward, *,
    gamma, actor_lr, critic_lr, terminated=False,
):
    if not isinstance(actor, GaussianWeightActor) or not isinstance(
        critic, SixParameterTDCritic
    ):
        raise TypeError("Expected GaussianWeightActor and SixParameterTDCritic")
    for name, v in (("actor_lr", actor_lr),("critic_lr",critic_lr)):
        if not np.isfinite(v) or float(v) <= 0:
            raise ValueError(f"{name} must be positive finite")
    if not isinstance(terminated, (bool,np.bool_)):
        raise ValueError("terminated must be bool (time-limit != termination)")
    value_now=critic.value(error)
    value_next=0. if terminated else critic.value(next_error)
    delta=float(td_error(reward,value_now,value_next,gamma))
    # BEFORE mutating either network; detached z and detached TD.
    loss, gradients=actor.loss_and_gradients(
        np.asarray(observation,float)[None,:],
        np.asarray(sampled_latent,float)[None,:],
        np.array([delta],float),
    )
    candidates={
        k:actor.params[k]-float(actor_lr)*g
        for k,g in gradients.items()
    }
    if any(not np.all(np.isfinite(x)) for x in candidates.values()):
        raise FloatingPointError("Actor candidate SGD weights nonfinite")
    result=critic.td_step(error,next_error,reward,gamma,critic_lr,
                          terminated=bool(terminated))
    if not np.isclose(result.td_error,delta,rtol=1e-12,atol=1e-12):
        raise RuntimeError("Actor and Critic TD mismatch")
    for key,updated in candidates.items():
        actor.params[key][...] = updated
    ag=float(np.sqrt(sum(np.sum(g*g) for g in gradients.values())))
    return TrainingStep(
        td=delta,actor_loss=float(loss),critic_loss=float(result.loss),
        actor_grad_norm=ag,actor_param_step_norm=float(actor_lr)*ag,
        critic_param_step_norm=float(np.linalg.norm(
            np.asarray(result.p_after)-np.asarray(result.p_before))),
        value_current=value_now,value_next=value_next,reward=float(reward),
    )
