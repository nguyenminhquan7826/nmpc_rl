"""On-policy rollout-block Actor update, with one frozen value snapshot.

The actor stays fixed while collecting a block. Every transition is used
once, with horizons block_length..1; the final partial block is flushed.
Critic can keep adapting live, but targets use the block-start P throughout.
"""
import numpy as np


def block_advantages(rewards, errors, next_error, p_snapshot, gamma, terminated):
    rewards=np.asarray(rewards,float)
    errors=np.asarray(errors,float)
    p=np.asarray(p_snapshot,float)
    if rewards.ndim!=1 or not len(rewards) or errors.shape!=(len(rewards),6):
        raise ValueError("Expected nonempty rewards and matching six-state errors")
    if p.shape!=(6,) or np.asarray(next_error).shape!=(6,):
        raise ValueError("Expected six-state terminal error and P")
    if not all(np.isfinite(v).all() for v in (rewards,errors,p,next_error)):
        raise ValueError("Nonfinite rollout values")
    if not 0<=gamma<=1:raise ValueError("Invalid gamma")
    tail=0. if terminated else -float(np.dot(p,np.asarray(next_error)**2))
    targets=np.empty(len(rewards))
    for i in range(len(rewards)-1,-1,-1):
        tail=rewards[i]+gamma*tail
        targets[i]=tail
    return targets+np.sum(errors**2*p,axis=1),targets


def update_actor_block(actor, observations, latents, advantages, config):
    """One SGD step on the MEAN loss; no implicit multiplication of LR by n."""
    loss,grad=actor.loss_and_gradients(
        np.asarray(observations),np.asarray(latents),np.asarray(advantages),
        output_gradient_scale=config.get('actor_output_lr_factors'),
        mu_regularization_coef=config.get('actor_mu_reg_coef',0.),
        mu_regularization_threshold=config.get('actor_mu_reg_threshold',2.))
    lr=float(config['actor_lr'])
    if not np.isfinite(lr) or lr<=0:raise ValueError('Actor LR must be positive')
    new={k:actor.params[k]-lr*g for k,g in grad.items()}
    if any(not np.isfinite(v).all() for v in new.values()):
        raise FloatingPointError('Nonfinite Actor update')
    for k,v in new.items():actor.params[k][...]=v
    return loss,float(lr*np.sqrt(sum(np.sum(g*g) for g in grad.values())))
