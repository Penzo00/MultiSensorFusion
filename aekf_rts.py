"""aekf_rts.py
Unified adaptive EKF + extended RTS smoother for a single 6-axis IMU.

EKF state (15):
    a   [0:3]   true linear acceleration in BODY frame, excluding gravity [m/s^2]
    w   [3:6]   true angular velocity in BODY frame                  [rad/s]
    ba  [6:9]   accelerometer bias                                   [m/s^2]
    bw  [9:12]  gyroscope bias                                       [rad/s]
    g   [12:15] gravity acceleration vector in BODY frame            [m/s^2]

Measurement models:
    gyro = w + bw + v_g
    acc  = a + ba - g + v_a
    ||g|| = G0 (optional 1-D pseudo-measurement)

No quaternion/attitude is tracked by the EKF.

The body-frame gravity dynamics are:
    g_dot = -w x g

After filtering + RTS smoothing, each sample is projected with a
    gravity-derived R_k = R(g_B(k)), recomputed from that sample's smoothed
    gravity vector, with zero yaw imposed as a coordinate convention. The
    same R_k projects both a_B(k) and w_B(k). Navigation is then reconstructed
    by integrating the projected a_E and w_E.

The RTS smoother is the "extended RTS" version used for the sequential
gyro-update -> gravity-propagation -> accelerometer-update pipeline:
    Phi[k] maps posterior state k-1 to the state immediately before
    the accelerometer update of sample k.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as Rot


I3 = np.eye(3)
N_X = 15

SL = {
    "a":  slice(0, 3),
    "w":  slice(3, 6),
    "ba": slice(6, 9),
    "bw": slice(9, 12),
    "g":  slice(12, 15),
}

IU = {3: np.triu_indices(3)}


def skew(v: np.ndarray) -> np.ndarray:
    """Return [v]_x such that [v]_x u = v x u."""
    v = np.asarray(v, float)
    return np.array([
        [0.0,  -v[2],  v[1]],
        [v[2],  0.0, -v[0]],
        [-v[1], v[0],  0.0],
    ])


def rk4_phi(A: np.ndarray, dt: float) -> np.ndarray:
    """RK4 polynomial map for x_dot = A x with constant A over dt."""
    n = A.shape[0]
    M = dt * A
    M2 = M @ M
    return (
        np.eye(n)
        + M
        + 0.5 * M2
        + (M2 @ M) / 6.0
        + (M2 @ M2) / 24.0
    )


@dataclass
class AEKFConfig:
    dt: float
    G0: float = 9.80665

    # Gauss-Markov time constants for a and w.
    tau_a: float = 2.0
    tau_w: float = 2.0

    # Initial state/covariance scales, all length 15:
    # [a(3), w(3), ba(3), bw(3), g(3)].
    P0_std: Optional[np.ndarray] = None

    # Adaptive process-variance vector, length 15:
    # a,w = piecewise-white-noise jump stds;
    # ba,bw,g = random-walk stds.
    Q0_std: Optional[np.ndarray] = None

    # Measurement stds [acc xyz, gyro xyz].
    R0_std: Optional[np.ndarray] = None

    # Optional gravity-magnitude pseudo-measurement ||g|| = G0.
    # Set to 0 to disable it.
    sigma_gnorm: float = 0.05

    # Adaptive noise.
    enable_adaptive: bool = True
    window: int = 100
    b: float = 0.98
    min_eig_floor: float = 1e-12

    # Stationary floors for adaptive process / measurement stds.
    Q_min_std: Optional[np.ndarray] = None   # length 15
    R_min_std: Optional[np.ndarray] = None   # length 6

    # Selective adaptation.  In dynamic experiments the gyro residual contains
    # real angular motion, so adapting R_gyr directly from that residual can
    # create a positive-feedback failure: large motion -> large R_gyr -> small
    # K_w -> w stays near zero -> still-large innovation.  Keep gyro R fixed by
    # default for dynamic runs, while still allowing acceleration/R and a,w Q
    # adaptation.
    adapt_gyro_R: bool = True
    adapt_bias_process: bool = True

    # Guards for the angular-rate process model.  These are jump standard
    # deviations in rad/s.  The minimum prevents a too-small imported prior
    # from making w effectively deterministic at zero.
    min_w_jump_std: float = 0.0
    max_w_jump_std: Optional[float] = None


class AEKFRTS:
    """15-state adaptive EKF with an extended RTS smoother."""

    def __init__(self, cfg: AEKFConfig, x0: np.ndarray):
        self.cfg = cfg
        self.x = np.asarray(x0, float).copy()

        if self.x.shape != (N_X,):
            raise ValueError(f"x0 must have shape ({N_X},), got {self.x.shape}")

        # Initial gravity is allowed to be slightly off-norm, but normalize
        # it when nonzero because gravity magnitude is known.
        gnorm = np.linalg.norm(self.x[SL["g"]])
        if gnorm > 0:
            self.x[SL["g"]] *= cfg.G0 / gnorm

        self.P = np.diag(self._check_vec(cfg.P0_std, N_X, "P0_std") ** 2)

        q0_std = self._check_vec(cfg.Q0_std, N_X, "Q0_std")
        self.q = q0_std ** 2
        if cfg.min_w_jump_std > 0.0:
            self.q[SL["w"]] = np.maximum(
                self.q[SL["w"]], cfg.min_w_jump_std ** 2
            )

        R0 = self._check_vec(cfg.R0_std, 6, "R0_std")
        self.R_acc = R0[0:3].copy() ** 2
        self.R_gyr = R0[3:6].copy() ** 2

        qmin = (
            np.zeros(N_X)
            if cfg.Q_min_std is None
            else self._check_vec(cfg.Q_min_std, N_X, "Q_min_std")
        )
        rmin = (
            np.zeros(6)
            if cfg.R_min_std is None
            else self._check_vec(cfg.R_min_std, 6, "R_min_std")
        )

        d_inf = 1.0 - cfg.b
        self.Q_min_sq = qmin * d_inf
        self.R_min_sq = rmin * d_inf

        self.fa = np.exp(-cfg.dt / cfg.tau_a)
        self.fw = np.exp(-cfg.dt / cfg.tau_w)

        self.g_earth = np.array([0.0, 0.0, -cfg.G0])

        self.hist_g = deque(maxlen=cfg.window)
        self.hist_a = deque(maxlen=cfg.window)
        self.step_i = 0

        # Arrays populated by step(..., record=True).
        self._xf = []
        self._Pf = []
        self._xp = []
        self._Pp = []
        self._Phi = []
        self._qhist = []

        # One-step linearization objects used by the composite transition.
        self._F1 = np.eye(N_X)
        self._IKH1 = np.eye(N_X)
        self._F3 = np.eye(N_X)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _check_vec(v, n, name):
        if v is None:
            raise ValueError(f"{name} must be supplied")
        a = np.asarray(v, float)
        if a.shape != (n,):
            raise ValueError(f"{name} must have shape ({n},), got {a.shape}")
        return a

    def sigmas(self) -> np.ndarray:
        """Current adaptive stds [s_a(3), s_w(3), s_ba(3), s_bw(3), s_g(3)]."""
        return np.sqrt(np.maximum(self.q, 0.0))

    def reset_recording(self):
        self._xf.clear()
        self._Pf.clear()
        self._xp.clear()
        self._Pp.clear()
        self._Phi.clear()
        self._qhist.clear()

    # ------------------------------------------------------------ process Q
    def build_Q(self) -> np.ndarray:
        """Closed-form PWN-style process covariance.

        The a/w jump noises also drive the body-frame gravity through
            g_dot = -w x g
        so the same angular-rate jump is correlated with g.
        """
        dt = self.cfg.dt
        g = self.x[SL["g"]]

        G = np.zeros((N_X, 6))

        # Acceleration jump.
        G[SL["a"], 0:3] = I3

        # Angular-rate jump.
        G[SL["w"], 3:6] = I3

        # First-order effect of angular-rate jump on gravity over dt.
        G[SL["g"], 3:6] = dt * skew(g)

        s2 = np.concatenate([
            self.q[SL["a"]],
            self.q[SL["w"]],
        ])

        Q = (G * s2) @ G.T

        # Independent random walks for biases and gravity-model mismatch.
        rw_idx = np.r_[6:15]
        Q[rw_idx, rw_idx] += self.q[rw_idx]

        return 0.5 * (Q + Q.T)

    # --------------------------------------------------------------- EKF step
    def step(self, acc: np.ndarray, gyro: np.ndarray, record: bool = True):
        """Process one IMU sample.

        Returns:
            x_f, P_f
        """
        acc = np.asarray(acc, float).reshape(3)
        gyro = np.asarray(gyro, float).reshape(3)

        # 1) predict a,w, biases + process Q
        self._predict()

        # 2) gyro update
        nu_g, K_g = self._update_gyro(gyro)

        # 3) propagate gravity using updated w
        self._propagate_gravity()

        if record:
            # Composite linearized map from posterior(k-1) to the
            # pre-accelerometer state of sample k.
            self._Phi.append(self._F3 @ self._IKH1 @ self._F1)
            self._xp.append(self.x.copy())
            self._Pp.append(self.P.copy())

        # 4) accelerometer + optional gravity-norm update
        nu_a, K_a = self._update_acc(acc)

        # 5) adaptive noise update
        self._adapt(nu_g, K_g, nu_a, K_a)

        self.step_i += 1

        if record:
            self._xf.append(self.x.copy())
            self._Pf.append(self.P.copy())
            self._qhist.append(self.q.copy())

        return self.x.copy(), self.P.copy()

    def run(self, acc: np.ndarray, gyro: np.ndarray, do_smooth: bool = True):
        """Run the full sequence and optionally RTS-smooth it."""
        acc = np.asarray(acc, float)
        gyro = np.asarray(gyro, float)
        if acc.ndim != 2 or acc.shape[1] != 3:
            raise ValueError("acc must have shape (N,3)")
        if gyro.shape != acc.shape:
            raise ValueError("gyro must have the same shape as acc")

        self.reset_recording()

        for k in range(len(acc)):
            self.step(acc[k], gyro[k], record=True)

        out = {
            "xf": np.asarray(self._xf),
            "Pf": np.asarray(self._Pf),
            "qhist": np.asarray(self._qhist),
        }

        if do_smooth:
            xs, Ps = self.smooth()
            out["xs"] = xs
            out["Ps"] = Ps

        return out

    # --------------------------------------------------------------- predict
    def _predict(self):
        x = self.x
        f = np.ones(N_X)
        f[SL["a"]] = self.fa
        f[SL["w"]] = self.fw

        F = np.diag(f)
        self._F1 = F

        Q = self.build_Q()

        x[SL["a"]] *= self.fa
        x[SL["w"]] *= self.fw

        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    # -------------------------------------------------------- generic update
    def _correct(self, nu: np.ndarray, H: np.ndarray, Rdiag: np.ndarray):
        PHt = self.P @ H.T
        S = H @ PHt
        S = 0.5 * (S + S.T)
        S[np.diag_indices_from(S)] += Rdiag

        # Solve S K^T = H P rather than explicitly inverting S.
        K = np.linalg.solve(S, PHt.T).T

        self.x += K @ nu

        # Joseph form keeps P positive semidefinite better than
        # P <- (I-KH)P alone.
        I_KH = np.eye(N_X) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + (K * Rdiag) @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return K

    # ------------------------------------------------------------- gyro meas
    def _update_gyro(self, gyro: np.ndarray):
        x = self.x

        nu = gyro - (x[SL["w"]] + x[SL["bw"]])

        H = np.zeros((3, N_X))
        H[:, SL["w"]] = I3
        H[:, SL["bw"]] = I3

        K = self._correct(nu, H, self.R_gyr)

        self._IKH1 = np.eye(N_X) - K @ H
        return nu, K

    # -------------------------------------------------------- gravity dyn
    def _propagate_gravity(self):
        """Propagate g_{k+1} = Exp(-[w]x dt) g_k using RK4 polynomial."""
        dt = self.cfg.dt
        x = self.x

        w = x[SL["w"]]
        g = x[SL["g"]]

        A = -skew(w)
        Phi_g = rk4_phi(A, dt)

        F = np.eye(N_X)
        F[SL["g"], SL["g"]] = Phi_g

        # d/dw_j of g^+ through the same second-order expansion used
        # in the lean quaternion/gravity implementation.
        for j in range(3):
            e = np.zeros(3)
            e[j] = 1.0
            Aj = -skew(e)

            F[SL["g"], 3 + j] = (
                dt * (Aj @ g)
                + 0.5 * dt * dt * (A @ (Aj @ g) + Aj @ (A @ g))
            )

        self._F3 = F

        x[SL["g"]] = Phi_g @ g

        self.P = F @ self.P @ F.T
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------- acc meas
    def _update_acc(self, acc: np.ndarray):
        x = self.x

        a = x[SL["a"]]
        ba = x[SL["ba"]]
        g = x[SL["g"]]

        # Main accelerometer measurement:
        #   acc = a + ba - g + v_a
        h = [a + ba - g]
        H_rows = []

        H0 = np.zeros((3, N_X))
        H0[:, SL["a"]] = I3
        H0[:, SL["ba"]] = I3
        H0[:, SL["g"]] = -I3
        H_rows.append(H0)

        # Optional known gravity magnitude ||g|| = G0.
        # This replaces the old quaternion-norm / gravity-attitude constraints.
        if self.cfg.sigma_gnorm > 0.0:
            gn = np.linalg.norm(g)
            if gn < 1e-12:
                # Degenerate only if initialization is pathological.
                h_g = self.cfg.G0
                Hg = np.zeros((1, N_X))
            else:
                h_g = gn
                Hg = np.zeros((1, N_X))
                Hg[0, SL["g"]] = g / gn

            h.append(np.array([h_g]))
            H_rows.append(Hg)

        h = np.concatenate(h)
        H = np.vstack(H_rows)

        z = np.zeros_like(h)
        z[0:3] = acc
        if self.cfg.sigma_gnorm > 0.0:
            z[3] = self.cfg.G0

        nu = z - h

        Rdiag = (
            np.concatenate([
                self.R_acc,
                [self.cfg.sigma_gnorm ** 2],
            ])
            if self.cfg.sigma_gnorm > 0.0
            else self.R_acc.copy()
        )

        K = self._correct(nu, H, Rdiag)
        return nu, K

    # ------------------------------------------------------------- adaptive
    def _adapt(
        self,
        nu_g: np.ndarray,
        K_g: np.ndarray,
        nu_a: np.ndarray,
        K_a: np.ndarray,
    ):
        cfg = self.cfg
        if not cfg.enable_adaptive:
            return

        # Only the physical 3-axis measurement residuals participate in
        # adaptive R; the gravity magnitude pseudo-measurement stays fixed.
        nu_a_phys = nu_a[:3]
        K_a_phys = K_a[:, :3]

        self.hist_g.append(nu_g)
        self.hist_a.append(nu_a_phys)

        Fg = np.mean(np.square(np.asarray(self.hist_g)), axis=0)
        Fa = np.mean(np.square(np.asarray(self.hist_a)), axis=0)

        d_k = (1.0 - cfg.b) / max(
            1.0 - cfg.b ** (self.step_i + 1),
            cfg.min_eig_floor,
        )

        temp = (
            np.einsum("ij,j,ij->i", K_g, Fg, K_g)
            + np.einsum("ij,j,ij->i", K_a_phys, Fa, K_a_phys)
        )

        old_q_std = np.sqrt(np.maximum(self.q, 0.0))
        target_q_std = np.sqrt(np.maximum(temp, 0.0))
        new_q_std = old_q_std.copy()

        # The physical fast states are a and w.  Bias random walks must not be
        # allowed to learn the deterministic dynamic motion.
        fast_idx = np.r_[np.arange(SL["a"].start, SL["a"].stop),
                         np.arange(SL["w"].start, SL["w"].stop)]
        new_q_std[fast_idx] = (
            (1.0 - d_k) * old_q_std[fast_idx]
            + d_k * target_q_std[fast_idx]
        )

        if cfg.adapt_bias_process:
            slow_idx = np.r_[np.arange(SL["ba"].start, SL["ba"].stop),
                             np.arange(SL["bw"].start, SL["bw"].stop),
                             np.arange(SL["g"].start, SL["g"].stop)]
            new_q_std[slow_idx] = (
                (1.0 - d_k) * old_q_std[slow_idx]
                + d_k * target_q_std[slow_idx]
            )

        # Apply configured floors and optional upper guard to angular-rate Q.
        if cfg.min_w_jump_std > 0.0:
            new_q_std[SL["w"]] = np.maximum(
                new_q_std[SL["w"]], cfg.min_w_jump_std
            )
        if cfg.max_w_jump_std is not None:
            new_q_std[SL["w"]] = np.minimum(
                new_q_std[SL["w"]], cfg.max_w_jump_std
            )

        new_q_std += self.Q_min_sq
        self.q = new_q_std ** 2

        # Adapt accelerometer measurement noise, but do not mistake dynamic
        # gyro innovations for changes in sensor noise unless explicitly asked.
        old_r_std = np.sqrt(np.maximum(np.concatenate([self.R_acc, self.R_gyr]), 0.0))
        target_r_std = np.sqrt(np.maximum(np.concatenate([Fa, Fg]), 0.0))
        new_r_std = old_r_std.copy()
        new_r_std[:3] = (
            (1.0 - d_k) * old_r_std[:3]
            + d_k * target_r_std[:3]
        )
        if cfg.adapt_gyro_R:
            new_r_std[3:6] = (
                (1.0 - d_k) * old_r_std[3:6]
                + d_k * target_r_std[3:6]
            )
        new_r_std += self.R_min_sq

        self.R_acc = new_r_std[:3] ** 2
        if cfg.adapt_gyro_R:
            self.R_gyr = new_r_std[3:6] ** 2

    # -------------------------------------------------------------- RTS
    @staticmethod
    def _psd_inverse(P: np.ndarray, eig_floor: float = 1e-10) -> np.ndarray:
        """Stable inverse for an approximately PSD covariance."""
        P = 0.5 * (P + P.T)
        s = np.sqrt(np.maximum(np.diag(P), 1e-300))
        Dinv = np.diag(1.0 / s)

        C = Dinv @ P @ Dinv
        C = 0.5 * (C + C.T)

        eig, V = np.linalg.eigh(C)
        Cinv = (V / np.maximum(eig, eig_floor)) @ V.T

        return Dinv @ Cinv @ Dinv

    def smooth(self, eig_floor: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
        """Run the extended RTS smoother on the recorded trajectory."""
        if len(self._xf) == 0:
            raise RuntimeError("No recorded trajectory. Call step(..., record=True) first.")

        xf = np.asarray(self._xf)
        Pf = np.asarray(self._Pf)
        xp = np.asarray(self._xp)
        Pp = np.asarray(self._Pp)
        Phi = np.asarray(self._Phi)

        n = len(xf)
        xs = xf.copy()
        Ps = Pf.copy()

        for k in range(n - 2, -1, -1):
            Pn = Pp[k + 1]
            Pinv = self._psd_inverse(Pn, eig_floor)

            C = Pf[k] @ Phi[k + 1].T @ Pinv

            xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
            Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pn) @ C.T
            Ps[k] = 0.5 * (Ps[k] + Ps[k].T)

        return xs, Ps

    # ----------------------------------------------------------- initialization
    @staticmethod
    def stationary_initial_state(
        acc_mean: np.ndarray,
        gyro_mean: np.ndarray,
        G0: float = 9.80665,
    ) -> np.ndarray:
        """Build a sensible x0 from a stationary interval."""
        acc_mean = np.asarray(acc_mean, float).reshape(3)
        gyro_mean = np.asarray(gyro_mean, float).reshape(3)

        n = np.linalg.norm(acc_mean)
        if n < 1e-12:
            raise ValueError("acc_mean is too small to infer gravity direction")

        # At rest: acc ≈ -g + ba, with true a = 0.
        g0 = -G0 * acc_mean / n

        x0 = np.zeros(N_X)
        x0[SL["a"]] = 0.0
        x0[SL["w"]] = 0.0
        x0[SL["ba"]] = acc_mean + g0
        x0[SL["bw"]] = gyro_mean
        x0[SL["g"]] = g0
        return x0

    # -------------------------------------------------------- postprocessing
    @staticmethod
    def _R_from_gravity(g_body: np.ndarray, G0: float) -> np.ndarray:
        """Body->Earth rotation from gravity, with yaw fixed to zero."""
        g = np.asarray(g_body, float).reshape(3)
        n = np.linalg.norm(g)
        if n < 1e-12:
            return np.eye(3)
        g = G0 * g / n
        gx, gy, gz = g
        h = max(np.hypot(gy, gz), 1e-15)
        pitch = np.arctan2(gx, h)
        roll = np.arctan2(-gy, -gz)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        return Ry @ Rx

    @staticmethod
    def reconstruct_navigation(
        xs: np.ndarray,
        dt: float,
        G0: float = 9.80665,
        v0: Optional[np.ndarray] = None,
        p0: Optional[np.ndarray] = None,
    ) -> dict:
        """Post-process a smoothed 15-state trajectory in the requested frame.

        At every sample k, R_k is recomputed from the smoothed body gravity
        g_B(k), with Earth-Z aligned to gravity and yaw=0 as the coordinate
        convention. The same R_k projects both a_B(k) and w_B(k).

        This intentionally does NOT integrate a quaternion/attitude state.
        The returned rotation matrices/quaternions are only a derived
        representation of the accumulated Earth-frame angular velocity.
        """
        xs = np.asarray(xs, float)
        if xs.ndim != 2 or xs.shape[1] != N_X:
            raise ValueError(f"xs must have shape (N,{N_X})")

        n = len(xs)
        R = np.empty((n, 3, 3))
        q_xyzw = np.empty((n, 4))
        a_earth = np.empty((n, 3))
        w_earth = np.empty((n, 3))
        theta = np.empty((n, 3))

        for k in range(n):
            R[k] = AEKFRTS._R_from_gravity(xs[k, SL["g"]], G0)
            a_earth[k] = R[k] @ xs[k, SL["a"]]
            w_earth[k] = R[k] @ xs[k, SL["w"]]

        # Angular accumulation is done only after tracking/smoothing.
        Racc = np.empty_like(R)
        Racc[0] = np.eye(3)
        theta[0] = 0.0
        for k in range(n - 1):
            w_mid = 0.5 * (w_earth[k] + w_earth[k + 1])
            dR = Rot.from_rotvec(w_mid * dt).as_matrix()
            Racc[k + 1] = dR @ Racc[k]
            theta[k + 1] = Rot.from_matrix(Racc[k + 1]).as_rotvec()
        q_xyzw[:] = Rot.from_matrix(Racc).as_quat()

        v = np.zeros((n, 3))
        p = np.zeros((n, 3))
        if v0 is not None:
            v[0] = np.asarray(v0, float).reshape(3)
        if p0 is not None:
            p[0] = np.asarray(p0, float).reshape(3)
        for k in range(n - 1):
            v[k + 1] = v[k] + 0.5 * dt * (a_earth[k] + a_earth[k + 1])
            p[k + 1] = p[k] + 0.5 * dt * (v[k] + v[k + 1])

        return {
            "R_gravity": R,
            "R": Racc,
            "q_xyzw": q_xyzw,
            "theta": theta,
            "a_earth": a_earth,
            "w_earth": w_earth,
            "v": v,
            "p": p,
        }


# ---------------------------------------------------------- Earth-frame blocks
def earth_state_covariance(
    xs: np.ndarray,
    Ps: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform [a_B, w_B, g_B] and their covariance into Earth frame.

    Returns:
        Y  : (N,9), concatenated [a_E, w_E, g_E_est]
        PY : (N,9,9), full covariance of those 9 quantities
    """
    xs = np.asarray(xs, float)
    Ps = np.asarray(Ps, float)
    R = np.asarray(R, float)

    if xs.ndim != 2 or xs.shape[1] != N_X:
        raise ValueError("xs must have shape (N,15)")
    if Ps.shape != (len(xs), N_X, N_X):
        raise ValueError("Ps must have shape (N,15,15)")
    if R.shape != (len(xs), 3, 3):
        raise ValueError("R must have shape (N,3,3)")

    Y = np.empty((len(xs), 9))
    PY = np.empty((len(xs), 9, 9))

    J = np.zeros((9, N_X))
    for k in range(len(xs)):
        Rk = R[k]
        Y[k] = np.concatenate([
            Rk @ xs[k, SL["a"]],
            Rk @ xs[k, SL["w"]],
            Rk @ xs[k, SL["g"]],
        ])
        J.fill(0.0)
        J[0:3, SL["a"]] = Rk
        J[3:6, SL["w"]] = Rk
        J[6:9, SL["g"]] = Rk
        PY[k] = J @ Ps[k] @ J.T

    return Y, PY