import numpy as np
import casadi as ca

from vehicle_model_casadi_variants import NX, NU, NP


def tracking_error(x, xref):
    """Full-state tracking error with wrapped heading error."""
    dpsi = x[2] - xref[2]
    epsi = ca.atan2(ca.sin(dpsi), ca.cos(dpsi))
    return ca.vertcat(
        x[0] - xref[0],
        x[1] - xref[1],
        epsi,
        x[3] - xref[3],
        x[4] - xref[4],
        x[5] - xref[5],
    )


def build_nmpc_solver(Fd, N, Q, R, Qf, cfg, backend='sqpmethod', adaptive=False):
    """
    Multiple-shooting NMPC with X0 treated as an NLP parameter.

    Decision variables:
        X[:,0] ... X[:,N-1]  <=> X_1 ... X_N
        U[:,0] ... U[:,N-1] <=> U_0 ... U_{N-1}

    NLP parameters:
        x0         : current state, shape (6,)
        u_prev     : previously applied input, shape (2,)
        xref_0_N   : reference states Xref_0 ... Xref_N, shape (6,N+1)
        p_model    : model parameters, shape (9,)
    """
    Q = ca.DM(Q)
    R = ca.DM(R)
    Qf = ca.DM(Qf)

    # X contains X_1 ... X_N; X_0 = x0 is a parameter.
    X = ca.SX.sym('X', NX, N)
    U = ca.SX.sym('U', NU, N)

    n_ref = NX * (N + 1)
    nP = NX + NU + n_ref + NP + (14 if adaptive else 0)
    P = ca.SX.sym('P', nP)

    off = 0
    x0 = P[off:off + NX]
    off += NX
    u_prev = P[off:off + NU]
    off += NU
    Xref = ca.reshape(P[off:off + n_ref], NX, N + 1)
    off += n_ref
    p_model = P[off:off + NP]
    off += NP
    if adaptive:
        Q = ca.diag(P[off:off+6])
        R = ca.diag(P[off+6:off+8])
        Qf = ca.diag(P[off+8:off+14])

    J = 0
    g_list = []
    lbg = []
    ubg = []

    du_min = np.asarray(cfg['du_min'], dtype=float)
    du_max = np.asarray(cfg['du_max'], dtype=float)

    beta_max = cfg.get('beta_max', None)
    vy_abs_max = cfg.get('vy_abs_max', None)

    x_prev = x0

    for i in range(N):
        x_i1 = X[:, i]      # X_{i+1|k}
        u_i = U[:, i]       # U_{i|k}

        # 1) Dynamics continuity constraint
        x_pred = Fd(x_prev, u_i, p_model)
        g_list.append(x_i1 - x_pred)
        lbg.extend([0.0] * NX)
        ubg.extend([0.0] * NX)

        # 2) Input-rate constraint
        if i == 0:
            du_i = u_i - u_prev
        else:
            du_i = u_i - U[:, i - 1]

        g_list.append(du_i)
        lbg.extend(du_min.tolist())
        ubg.extend(du_max.tolist())

        # 3) Optional small-slip / lateral-speed constraints:
        #    |vy| <= beta_max * vx
        #    |vy| <= vy_abs_max
        if beta_max is not None:
            vx_i = x_i1[0]
            vy_i = x_i1[1]
            g_list.append(ca.vertcat(
                vy_i - beta_max * vx_i,
                -vy_i - beta_max * vx_i,
            ))
            lbg.extend([-ca.inf, -ca.inf])
            ubg.extend([0.0, 0.0])

        if vy_abs_max is not None:
            vy_i = x_i1[1]
            g_list.append(ca.vertcat(vy_i, -vy_i))
            lbg.extend([-ca.inf, -ca.inf])
            ubg.extend([vy_abs_max, vy_abs_max])

        # 4) Cost: X_1...X_{N-1} use Q; X_N uses Qf.
        e_i1 = tracking_error(x_i1, Xref[:, i + 1])
        if i < N - 1:
            J += ca.mtimes([e_i1.T, Q, e_i1])
        else:
            J += ca.mtimes([e_i1.T, Qf, e_i1])

        J += ca.mtimes([du_i.T, R, du_i])
        x_prev = x_i1

    # Decision vector: [vec(X_1...X_N); vec(U_0...U_{N-1})]
    z = ca.vertcat(
        ca.reshape(X, NX * N, 1),
        ca.reshape(U, NU * N, 1),
    )

    g = ca.vertcat(*g_list)
    nlp = {'x': z, 'f': J, 'g': g, 'p': P}

    if backend == 'sqpmethod':
        opts = {
            'qpsol': 'qrqp',
            'max_iter': int(cfg.get('sqp_max_iter', 5)),
            'print_header': False,
            'print_iteration': False,
            'print_status': False,
            'print_time': False,
            'qpsol_options': {
                'print_header': False,
                'print_iter': False,
            },
        }
    elif backend == 'ipopt':
        opts = {
            'ipopt.print_level': 0,
            'print_time': False,
            'ipopt.max_iter': int(cfg.get('ipopt_max_iter', 100)),
        }
    else:
        raise ValueError("backend must be 'sqpmethod' or 'ipopt'")

    solver = ca.nlpsol('nmpc_solver', backend, nlp, opts)

    # Variable bounds. State/input hard bounds are encoded as lbx/ubx.
    state_lb = np.array([
        cfg['vx_min'],
        cfg.get('vy_min', -np.inf),
        cfg.get('psi_min', -np.inf),
        -cfg['r_max'],
        cfg.get('X_min', -np.inf),
        cfg.get('Y_min', -np.inf),
    ], dtype=float)

    state_ub = np.array([
        cfg['vx_max'],
        cfg.get('vy_max', np.inf),
        cfg.get('psi_max', np.inf),
        cfg['r_max'],
        cfg.get('X_max', np.inf),
        cfg.get('Y_max', np.inf),
    ], dtype=float)

    input_lb = np.asarray(cfg['u_min'], dtype=float)
    input_ub = np.asarray(cfg['u_max'], dtype=float)

    lbx = np.concatenate([
        np.tile(state_lb, N),
        np.tile(input_lb, N),
    ])

    ubx = np.concatenate([
        np.tile(state_ub, N),
        np.tile(input_ub, N),
    ])

    return {
        'solver': solver,
        'lbx': lbx,
        'ubx': ubx,
        'lbg': np.asarray(lbg, dtype=float),
        'ubg': np.asarray(ubg, dtype=float),
        'N': N,
        'n_decision': NX * N + NU * N,
        'n_parameter': nP,
        'n_constraint': int(g.numel()),
    }


def pack_parameters(x0, u_prev, xref, p_model, weights=None):
    """
    xref input shape: (N+1, 6), one state per row.
    Packing matches CasADi reshape(..., 6, N+1), which is column-major.
    """
    x0 = np.asarray(x0, dtype=float).reshape(NX)
    u_prev = np.asarray(u_prev, dtype=float).reshape(NU)
    xref = np.asarray(xref, dtype=float)
    p_model = np.asarray(p_model, dtype=float).reshape(NP)

    if xref.ndim != 2 or xref.shape[1] != NX:
        raise ValueError('xref must have shape (N+1, 6)')

    xref_flat = xref.T.reshape(-1, order='F')
    parts = [x0, u_prev, xref_flat, p_model]
    if weights is not None:
        w = np.concatenate([np.asarray(v, dtype=float).reshape(-1) for v in weights])
        if w.size != 14 or not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("weights must contain 6+2+6 positive finite diagonal entries")
        parts.append(w)
    return np.concatenate(parts)


def unpack_solution(z_star, N):
    z_star = np.asarray(z_star, dtype=float).reshape(-1)
    nX = NX * N

    X_star = z_star[:nX].reshape((NX, N), order='F')
    U_star = z_star[nX:].reshape((NU, N), order='F')
    return X_star, U_star


def make_initial_guess(x0, u_prev, N):
    """Simple initial guess; later replace with shifted warm start."""
    x0 = np.asarray(x0, dtype=float).reshape(NX)
    u_prev = np.asarray(u_prev, dtype=float).reshape(NU)

    X_guess = np.tile(x0.reshape(NX, 1), (1, N))
    U_guess = np.tile(u_prev.reshape(NU, 1), (1, N))

    return np.concatenate([
        X_guess.reshape(-1, order='F'),
        U_guess.reshape(-1, order='F'),
    ])


def shift_warm_start(X_star, U_star):
    """Shift a converged solution to initialize the next sampling instant."""
    N = U_star.shape[1]

    X_guess = np.empty_like(X_star)
    U_guess = np.empty_like(U_star)

    if N > 1:
        X_guess[:, :-1] = X_star[:, 1:]
        U_guess[:, :-1] = U_star[:, 1:]

    X_guess[:, -1] = X_star[:, -1]
    U_guess[:, -1] = U_star[:, -1]

    return np.concatenate([
        X_guess.reshape(-1, order='F'),
        U_guess.reshape(-1, order='F'),
    ])
