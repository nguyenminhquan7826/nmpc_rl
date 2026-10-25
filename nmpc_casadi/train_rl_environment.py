"""Live Actor/Critic/NMPC transition environment on the pinned TRAIN track.

The NMPC solver is a black box: no gradient passes through it.
Dynamics/constraints and two-attempt fallback follow the existing
fixed-NMPC simulator. This environment makes no stability theorem claim.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np

import baseline as b
from nmpc_casadi import (
    build_nmpc_solver, unpack_solution, shift_warm_start,
)
from critic_nmpc_bridge import actor_critic_nmpc_parameters
from direct_p_critic import error_from_state
from gaussian_weight_actor import build_actor_observation
from rl_paper_core import paper_reward
from gaussian_state_sensor import GaussianStateSensor


@dataclass(frozen=True)
class Observation:
    features: np.ndarray
    error: np.ndarray
    physical: np.ndarray
    reference: dict
    true_physical: np.ndarray | None = None
    true_error: np.ndarray | None = None
    true_reference: dict | None = None
    measurement_noise: np.ndarray | None = None


class TrainEnvironment:
    """A single TRAIN episode; caller stages TRAIN path before creating it."""

    def __init__(self, *, train_length_m, q_initial, r_initial, p_initial,
                 task_qe, task_re, max_steps, ey, heading_deg, sensor=None):
        if sensor is not None and not isinstance(sensor, GaussianStateSensor):
            raise TypeError("sensor must be GaussianStateSensor or None")
        self.sensor=sensor
        self.train_length_m=float(train_length_m)
        if not np.isfinite(self.train_length_m) or self.train_length_m<=0:
            raise ValueError("train_length_m must be positive finite")
        self.task_qe=np.asarray(task_qe,dtype=float).copy()
        self.task_re=np.asarray(task_re,dtype=float).copy()
        if self.task_qe.shape!=(6,) or self.task_re.shape!=(2,) or (
            np.any(self.task_qe<=0) or np.any(self.task_re<=0) or
            not np.all(np.isfinite(self.task_qe)) or
            not np.all(np.isfinite(self.task_re))
        ):
            raise ValueError("positive fixed reward Qe[6], Re[2] required")
        self.max_steps=int(max_steps)
        if self.max_steps<1 or self.max_steps!=max_steps:
            raise ValueError("max_steps must be a positive integer")
        self.ey=float(ey)
        self.heading_deg=float(heading_deg)
        if not np.isfinite(self.ey) or not np.isfinite(self.heading_deg):
            raise ValueError("initial offsets must be finite")
        self.solver=build_nmpc_solver(
            b.Fd_controller,b.N,np.diag(q_initial),np.diag(r_initial),
            np.diag(p_initial),b.cfg,adaptive_cost_weights=True)
        self.reset()

    def reset(self):
        self.rm=b.make_reference_manager()
        # Independent progress trackers: noisy localization must not alter
        # the true path-projection metrics or termination criterion.
        self.true_rm=(self.rm if self.sensor is None else b.make_reference_manager())
        self.x,self.prev,self.guess,_=b.build_initial_state_and_guess(
            self.rm,self.solver,self.ey,np.deg2rad(self.heading_deg))
        self.k=0
        self.fail_streak=0
        self.done=False
        return self.observe()

    def observe(self):
        # The plant is always true; every controller observes a noisy measurement.
        if self.sensor is None:
            measured=self.x.copy()
            noise=np.zeros(6)
        else:
            measured,noise=self.sensor.measure(self.x,self.k)
        ref=self.rm.get_nmpc_reference(
            float(measured[4]),float(measured[5]),b.Ts,b.N)
        true_ref=(ref if self.sensor is None else self.true_rm.get_nmpc_reference(
            float(self.x[4]),float(self.x[5]),b.Ts,b.N))
        er=error_from_state(measured,ref["xref"][0])
        true_er=error_from_state(self.x,true_ref["xref"][0])
        curvature=self.rm.interpolate_curvature(ref["s_horizon"])
        features=build_actor_observation(
            measured,ref["xref"][0],self.prev,
            path_progress_m=float(ref["projection"]["s"]),
            path_start_m=float(self.rm.s_start),path_end_m=float(b.CAD_S_END),
            curvature_now=float(curvature[0]),
            curvature_horizon_end=float(curvature[-1]),
            reference_origin_xy=np.array([self.rm.X[0],self.rm.Y[0]]),
            reference_initial_heading_rad=float(self.rm.psi[0]),
            train_path_length_m=self.train_length_m)
        return Observation(features,er,measured,ref,self.x.copy(),true_er,true_ref,noise)

    def step(self,actor_q,actor_r,critic,pre):
        """Apply actual first NMPC input and return (next_obs,reward,terminated,
        truncated,step_diagnostics); includes paper Eq. (4) only.
        """
        if self.done:
            raise RuntimeError("Episode finished; call reset")
        x0=self.x.copy()
        prev=self.prev.copy()
        params=actor_critic_nmpc_parameters(
            pre.physical,prev,pre.reference["xref"],b.p_nominal,
            actor_q,actor_r,critic)
        success=False
        primary=False
        violation=float("inf")
        attempts=0
        solve_start=time.perf_counter()
        for attempt in range(2):
            attempts+=1
            guess=self.guess if attempt==0 else b.build_fresh_dynamics_rollout_guess(
                self.rm,pre.physical,self.prev,pre.reference)
            try:
                raw=self.solver["solver"](
                    x0=guess,p=params,
                    lbx=self.solver["lbx"],ubx=self.solver["ubx"],
                    lbg=self.solver["lbg"],ubg=self.solver["ubg"])
                solution=np.asarray(raw["x"],float).reshape(-1)
                constraint=np.asarray(raw["g"],float).reshape(-1)
                violation=float(max(
                    np.max(np.maximum(self.solver["lbx"]-solution,0)),
                    np.max(np.maximum(solution-self.solver["ubx"],0)),
                    np.max(np.maximum(self.solver["lbg"]-constraint,0)),
                    np.max(np.maximum(constraint-self.solver["ubg"],0))))
                success=bool(self.solver["solver"].stats().get("success",False)
                         and np.all(np.isfinite(solution))
                         and np.isfinite(violation) and violation<=1e-6)
                if success:
                    xpred, upred=unpack_solution(solution,b.N)
                    u=upred[:,0].copy()
                    self.guess=shift_warm_start(xpred,upred)
                    primary=attempt==0
                    break
            except Exception:
                success=False
        elapsed_ms=1000.*(time.perf_counter()-solve_start)
        if not success:
            u=prev.copy()
            self.fail_streak+=1
        else:
            self.fail_streak=0
        du=u-prev
        # RL only has the observation-derived reward; evaluation uses true reward.
        reward=float(paper_reward(pre.error,du,self.task_qe,self.task_re))
        reward_true=float(paper_reward(pre.true_error,du,self.task_qe,self.task_re))
        candidate_x=np.asarray(b.Fd_plant(self.x,u,b.p_nominal),float).reshape(-1)
        invalid=candidate_x.shape!=(6,) or not np.all(np.isfinite(candidate_x))
        if not invalid:
            self.x=candidate_x
        self.prev=u.copy()
        self.k+=1
        next_obs=self.observe()
        reached=bool(next_obs.true_reference["projection"]["s"]>=b.CAD_S_END-.03)
        catastrophic=bool(invalid or self.fail_streak>=3)
        terminated=bool(reached or catastrophic)
        truncated=bool(self.k>=self.max_steps and not terminated)
        self.done=terminated or truncated
        return next_obs,reward,terminated,truncated,{
            "step":self.k,"solver_success":success,
            "solve_ms":float(elapsed_ms),
            "deadline_missed":bool(elapsed_ms>b.DEADLINE_MS),
            "solver_primary_success":primary,"solver_attempts":attempts,
            "max_constraint_violation":violation,
            "invalid_plant_state":invalid,
            "failure_streak":self.fail_streak,
            "reached_end":reached,"failure_terminal":catastrophic,
            "progress_m":float(next_obs.true_reference["projection"]["s"]),
            "cte_m":float(pre.true_reference["projection"]["distance"]),
            "reward_true":reward_true,"reward_observed":reward,
            "cte_measured_m":float(pre.reference["projection"]["distance"]),
            "measurement_noise_norm":float(np.linalg.norm(pre.measurement_noise)),
            **{f"sensor_noise_{i}":float(v) for i,v in enumerate(pre.measurement_noise)},
            **{f"state_measured_{i}":float(v) for i,v in enumerate(pre.physical)},
            **{f"state_true_{i}":float(v) for i,v in enumerate(x0)},
            # Log physical XY directly, aligned with the pre-action error.
            # Keep the final post-action point so plots include the endpoint.
            "t_s":float((self.k-1)*b.Ts),
            "X_m":float(x0[4]),"Y_m":float(x0[5]),
            "X_ref_m":float(pre.true_reference["xref"][0,4]),
            "Y_ref_m":float(pre.true_reference["xref"][0,5]),
            "X_next_m":float(self.x[4]),"Y_next_m":float(self.x[5]),
            "du_a":float(du[0]),"du_delta":float(du[1]),
            "a":float(u[0]),"steering":float(u[1]),
        }
