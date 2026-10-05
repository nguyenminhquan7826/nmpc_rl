import numpy as np
import pandas as pd


class ReferenceManager:
    """
    Reference manager for an OPEN path parameterized by arc length s.

    Input database columns:
        s_m
        vx_ref_mps
        vy_ref_mps
        psi_ref_rad
        r_ref_radps
        X_ref_m
        Y_ref_m
        kappa_1_per_m

    NMPC state order:
        [vx, vy, psi, r, X, Y]

    Main jobs:
      1) Project the current localization (X, Y) onto the CAD reference path
         to estimate path progress s*.
      2) Build N+1 future reference states on the NMPC horizon.
      3) Interpolate all reference variables as functions of s.
    """

    def __init__(
        self,
        csv_path,
        enforce_monotonic=True,
        search_back_m=0.25,
        search_forward_m=2.0,
        stop_at_end=True,
        use_midpoint_progress=True,
    ):
        self.df = pd.read_csv(csv_path)

        required = [
            "s_m",
            "vx_ref_mps",
            "vy_ref_mps",
            "psi_ref_rad",
            "r_ref_radps",
            "X_ref_m",
            "Y_ref_m",
            "kappa_1_per_m",
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.s = self.df["s_m"].to_numpy(dtype=float)
        self.vx = self.df["vx_ref_mps"].to_numpy(dtype=float)
        self.vy = self.df["vy_ref_mps"].to_numpy(dtype=float)
        self.psi = np.unwrap(
            self.df["psi_ref_rad"].to_numpy(dtype=float)
        )
        self.r = self.df["r_ref_radps"].to_numpy(dtype=float)
        self.X = self.df["X_ref_m"].to_numpy(dtype=float)
        self.Y = self.df["Y_ref_m"].to_numpy(dtype=float)
        self.kappa = self.df["kappa_1_per_m"].to_numpy(dtype=float)

        if len(self.s) < 2:
            raise ValueError("Reference path must contain at least 2 points.")

        if np.any(np.diff(self.s) <= 0.0):
            raise ValueError("s_m must be strictly increasing.")

        self.s_start = float(self.s[0])
        self.s_end = float(self.s[-1])

        self.enforce_monotonic = bool(enforce_monotonic)
        self.search_back_m = float(search_back_m)
        self.search_forward_m = float(search_forward_m)
        self.stop_at_end = bool(stop_at_end)
        self.use_midpoint_progress = bool(use_midpoint_progress)

        # Optional time parameterization exported with the new Xref file.
        self.t_ref = (
            self.df["t_ref_s"].to_numpy(dtype=float)
            if "t_ref_s" in self.df.columns else None
        )

        # Segment data for geometric projection
        self.seg_x0 = self.X[:-1]
        self.seg_y0 = self.Y[:-1]
        self.seg_dx = self.X[1:] - self.X[:-1]
        self.seg_dy = self.Y[1:] - self.Y[:-1]
        self.seg_ds = self.s[1:] - self.s[:-1]
        self.seg_len2 = self.seg_dx**2 + self.seg_dy**2

        self.prev_s = None
        self.prev_seg_idx = None

    def reset(self, s0=None):
        """Reset progress memory, e.g. before a new run."""
        if s0 is None:
            self.prev_s = None
            self.prev_seg_idx = None
        else:
            self.prev_s = float(
                np.clip(s0, self.s_start, self.s_end)
            )
            self.prev_seg_idx = int(
                np.clip(
                    np.searchsorted(self.s, self.prev_s) - 1,
                    0,
                    len(self.s) - 2,
                )
            )

    def _candidate_segment_indices(self):
        """
        Use a local progress window after initialization to reduce the risk
        of jumping to a geometrically nearby but topologically distant branch.
        """
        nseg = len(self.s) - 1

        if self.prev_s is None:
            return np.arange(nseg, dtype=int)

        s_lo = max(self.s_start, self.prev_s - self.search_back_m)
        s_hi = min(self.s_end, self.prev_s + self.search_forward_m)

        mask = (self.s[:-1] <= s_hi) & (self.s[1:] >= s_lo)
        idx = np.flatnonzero(mask)

        if len(idx) == 0:
            return np.arange(nseg, dtype=int)

        return idx

    def project_xy_to_path(self, X_vehicle, Y_vehicle):
        """
        Orthogonally project localization point (X_vehicle, Y_vehicle)
        onto the nearest eligible path segment.

        Returns:
            dict with:
                s          : path progress [m]
                X_proj     : projected X [m]
                Y_proj     : projected Y [m]
                distance   : Euclidean cross-track distance magnitude [m]
                segment_idx
                alpha      : segment interpolation factor in [0,1]
        """
        px = float(X_vehicle)
        py = float(Y_vehicle)

        idx = self._candidate_segment_indices()

        x0 = self.seg_x0[idx]
        y0 = self.seg_y0[idx]
        dx = self.seg_dx[idx]
        dy = self.seg_dy[idx]
        len2 = self.seg_len2[idx]

        # Projection factor on each candidate segment
        safe_len2 = np.maximum(len2, 1e-15)
        alpha = ((px - x0) * dx + (py - y0) * dy) / safe_len2
        alpha = np.clip(alpha, 0.0, 1.0)

        xproj = x0 + alpha * dx
        yproj = y0 + alpha * dy

        dist2 = (px - xproj)**2 + (py - yproj)**2
        j_local = int(np.argmin(dist2))
        j = int(idx[j_local])

        a = float(alpha[j_local])

        s_proj = float(
            self.s[j] + a * (self.s[j + 1] - self.s[j])
        )

        # For a forward-only route, do not allow progress to jump backward
        # because of small localization noise.
        if (
            self.enforce_monotonic
            and self.prev_s is not None
            and s_proj < self.prev_s
        ):
            s_proj = self.prev_s

            # Re-evaluate geometry at the clamped progress value
            xproj = self._interp_scalar(self.X, s_proj)
            yproj = self._interp_scalar(self.Y, s_proj)
            distance = float(
                np.hypot(px - xproj, py - yproj)
            )
        else:
            xproj = float(xproj[j_local])
            yproj = float(yproj[j_local])
            distance = float(np.sqrt(dist2[j_local]))

        self.prev_s = s_proj
        self.prev_seg_idx = j

        return {
            "s": s_proj,
            "X_proj": xproj,
            "Y_proj": yproj,
            "distance": distance,
            "segment_idx": j,
            "alpha": a,
        }

    def _interp_scalar(self, values, s_query):
        return float(np.interp(s_query, self.s, values))

    def interpolate_reference(self, s_query):
        """
        Interpolate full state reference at one or many s values.

        Returns shape:
            scalar s -> (6,)
            array s  -> (M,6)

        State order:
            [vx, vy, psi, r, X, Y]
        """
        sq = np.asarray(s_query, dtype=float)
        sq_clip = np.clip(sq, self.s_start, self.s_end)

        vx = np.interp(sq_clip, self.s, self.vx)
        vy = np.interp(sq_clip, self.s, self.vy)
        psi = np.interp(sq_clip, self.s, self.psi)
        r = np.interp(sq_clip, self.s, self.r)
        X = np.interp(sq_clip, self.s, self.X)
        Y = np.interp(sq_clip, self.s, self.Y)

        # Open-path terminal handling:
        # once reference progress reaches the endpoint, request zero speed
        # and zero yaw rate rather than asking the vehicle to keep moving
        # while the position reference is frozen.
        if self.stop_at_end:
            reached_end = sq >= (self.s_end - 1e-10)
            vx = np.where(reached_end, 0.0, vx)
            vy = np.where(reached_end, 0.0, vy)
            r = np.where(reached_end, 0.0, r)

        out = np.stack([vx, vy, psi, r, X, Y], axis=-1)

        if np.ndim(s_query) == 0:
            return out.reshape(6)

        return out

    def interpolate_time(self, s_query):
        """Interpolate the optional t_ref(s) column from the Xref file."""
        if self.t_ref is None:
            raise ValueError("Reference file does not contain t_ref_s")
        sq = np.asarray(s_query, dtype=float)
        sq_clip = np.clip(sq, self.s_start, self.s_end)
        out = np.interp(sq_clip, self.s, self.t_ref)
        if np.ndim(s_query) == 0:
            return float(out)
        return out

    def interpolate_curvature(self, s_query):
        sq = np.asarray(s_query, dtype=float)
        sq_clip = np.clip(sq, self.s_start, self.s_end)
        out = np.interp(sq_clip, self.s, self.kappa)

        if np.ndim(s_query) == 0:
            return float(out)

        return out

    def build_horizon_from_s(self, s0, Ts, N):
        """
        Build N+1 NMPC reference states.

        Progress model:
            s_{i+1} = s_i + v_path_ref(s_i) * Ts

        where
            v_path_ref = sqrt(vx_ref^2 + vy_ref^2)

        For the current baseline vy_ref=0, this reduces to vx_ref.
        """
        Ts = float(Ts)
        N = int(N)

        s_h = np.zeros(N + 1, dtype=float)
        s_h[0] = float(np.clip(s0, self.s_start, self.s_end))

        for i in range(N):
            # Progress dynamics: ds/dt = v_ref(s).
            # With the new Xref, v_ref varies with curvature.  A midpoint
            # step is more accurate than a single forward-Euler step.
            ref_i = self.interpolate_reference(s_h[i])
            v_i = float(np.hypot(ref_i[0], ref_i[1]))
            v_i = max(v_i, 0.0)

            if self.use_midpoint_progress and s_h[i] < self.s_end:
                s_mid = min(s_h[i] + 0.5 * Ts * v_i, self.s_end)
                ref_mid = self.interpolate_reference(s_mid)
                v_mid = float(np.hypot(ref_mid[0], ref_mid[1]))
                v_step = max(v_mid, 0.0)
            else:
                v_step = v_i

            s_h[i + 1] = min(
                s_h[i] + v_step * Ts,
                self.s_end,
            )

        xref_h = self.interpolate_reference(s_h)

        return {
            "s_horizon": s_h,
            "xref": xref_h,
        }

    def get_nmpc_reference(self, X_vehicle, Y_vehicle, Ts, N):
        """
        Main runtime call.

        1) Project current localization to path -> s*
        2) Build N+1 reference states from s*

        Returns:
            projection : dict
            s_horizon  : shape (N+1,)
            xref       : shape (N+1,6)
        """
        projection = self.project_xy_to_path(
            X_vehicle,
            Y_vehicle,
        )

        horizon = self.build_horizon_from_s(
            projection["s"],
            Ts,
            N,
        )

        return {
            "projection": projection,
            "s_horizon": horizon["s_horizon"],
            "xref": horizon["xref"],
        }
