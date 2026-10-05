import casadi as ca

NX = 6
NU = 2
NP = 9

# State order: [vx, vy, psi, r, X, Y]
# Input order: [a, delta]
# Parameter order: [m, Iz, Cf, Cr, lf, lr, mu, g, eps_v]


def build_continuous_model(rear_force_convention='restoring'):
    """
    Build the nonlinear dynamic bicycle model.

    rear_force_convention:
      - 'paper':     F_yr = +Cr*(vy-lr*r)/vx_safe, exactly as Eq. (16)-(17)
                     is typeset in the supplied paper.
      - 'restoring': F_yr = -Cr*(vy-lr*r)/vx_safe, the standard restoring-force
                     convention for a positive rear cornering stiffness.

    The first lane-change closed-loop test showed that the paper-typed convention
    is laterally unstable with the paper's positive Cr; therefore the working
    baseline uses 'restoring' while preserving 'paper' for reproduction checks.
    """
    if rear_force_convention not in ('paper', 'restoring'):
        raise ValueError("rear_force_convention must be 'paper' or 'restoring'")

    x = ca.SX.sym('x', NX)
    u = ca.SX.sym('u', NU)
    p = ca.SX.sym('p', NP)

    vx, vy, psi, r, Xg, Yg = [x[i] for i in range(NX)]
    a, delta = u[0], u[1]
    m, Iz, Cf, Cr, lf, lr, mu, g, eps_v = [p[i] for i in range(NP)]

    vx_safe = ca.sqrt(vx * vx + eps_v * eps_v)
    theta_f = (vy + lf * r) / vx_safe
    theta_r = (vy - lr * r) / vx_safe

    F_yf = Cf * (delta - theta_f)
    sign_r = 1.0 if rear_force_convention == 'paper' else -1.0
    F_yr = sign_r * Cr * theta_r

    vx_dot = a - (F_yf / m) * ca.sin(delta) - mu * g + vy * r
    vy_dot = (F_yr + F_yf * ca.cos(delta)) / m - vx * r
    psi_dot = r
    r_dot = (lf * F_yf * ca.cos(delta) - lr * F_yr) / Iz
    X_dot = vx * ca.cos(psi) - vy * ca.sin(psi)
    Y_dot = vx * ca.sin(psi) + vy * ca.cos(psi)

    xdot = ca.vertcat(vx_dot, vy_dot, psi_dot, r_dot, X_dot, Y_dot)
    return ca.Function(
        f'f_cont_{rear_force_convention}', [x, u, p], [xdot],
        ['x', 'u', 'p'], ['xdot']
    )


def build_rk4_discrete_model(f_cont, Ts, name='F_rk4'):
    x = ca.SX.sym('x', NX)
    u = ca.SX.sym('u', NU)
    p = ca.SX.sym('p', NP)

    # ZOH: u is identical in k1...k4 over this sample interval.
    k1 = f_cont(x, u, p)
    k2 = f_cont(x + 0.5 * Ts * k1, u, p)
    k3 = f_cont(x + 0.5 * Ts * k2, u, p)
    k4 = f_cont(x + Ts * k3, u, p)
    x_next = x + (Ts / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return ca.Function(name, [x, u, p], [x_next], ['x', 'u', 'p'], ['x_next'])
