#!/usr/bin/env python3
"""
run_general_multisensor_dynamic.py
==================================

General multisensor pipeline for CSV files containing an arbitrary number of
6-axis IMUs named s1, s2, ..., sN.

Default input directory:
    DATA/                       (next to this script)

Expected columns for every sensor i:
    si_acc_x [mg], si_acc_y [mg], si_acc_z [mg]
    si_gyro_x [dps], si_gyro_y [dps], si_gyro_z [dps]

Other CSV columns are ignored (e.g. time1/time2/time3 metadata columns).

PIPELINE FOR EACH CSV FILE
--------------------------
1) Discover s1...sN automatically from the column names.
2) Run one 15-state adaptive EKF + first RTS smoother per IMU:
       x_B = [a_B, w_B, b_a, b_w, g_B].
3) Estimate constant sensor->rigid-body rotations C_i by a COVARIANCE-WEIGHTED
   Wahba problem on the joint [g, w] smoothed estimates: residuals are weighted
   by the full 6x6 covariance P_s1 + C P_si C^T (weighted-SVD start, then
   Gauss-Newton on SO(3)), optionally refined leave-one-out against the fused
   other sensors.  s1 defines the reference rigid-body frame.  Gravity alone
   cannot fix yaw; the gyro term resolves it when excited, and its weight is set
   by its covariance, not by an ad-hoc factor.  The registration covariance
   (yaw uncertainty) is reported.
4) Reconstruct a common rigid-body attitude after filtering only (the [g,w]
   consensus is the information-form, covariance-weighted fusion, no medians):
       - yaw(t=0) = 0 convention;
       - common angular rate drives the attitude;
       - each IMU's own smoothed gravity is used at every sample to correct
         tilt without resetting yaw.
5) Project every IMU, sample by sample, to Earth XYZ:
       a_E, w_E, b_a,E, b_w,E, g_E.
6) Fuse a_E and w_E across all IMUs by full 6x6 covariance-weighted
   information fusion at every timestep.
7) Optionally apply a second 6-state RTS smoother to the fused [a_E,w_E].
   This is a temporal regularizer, not an exact second Bayesian smoother,
   because per-IMU RTS already induces temporal correlations and those
   cross-time covariances are not stored.
8) Reconstruct velocity, position, and angular position only after the final
   acceleration/angular-rate history is selected.
9) Plot all requested quantities in Earth XYZ.
10) Save per-file NPZ/CSV diagnostics.

IMPORTANT PHYSICAL ASSUMPTIONS
------------------------------
- Samples across sensors in a file are synchronized (same row = same time).
- Relative sensor orientations are constant during a file.
- The sensors measure the same rigid-body angular velocity.
- The acceleration fusion assumes coincident sensor origins, or negligible
  lever-arm effects. For non-negligible offsets r_i, translational acceleration
  differs by alpha x r_i + omega x (omega x r_i), and that correction must be
  added before acceleration fusion.
- Gravity provides tilt, but not absolute yaw.  yaw(t=0)=0 is a coordinate
  convention; dynamic gyro information is used to propagate yaw afterwards.

This script does NOT fuse different CSV files together.  Each CSV file is a
separate experiment / trajectory and its own s1...sN are fused internally.

Dependencies:
    aekf_rts.py
    preferably the existing ekf_run.py with make_cfg(acc, gyro), so your
    established noise priors are reused.  A conservative fallback config is
    provided if ekf_run.py is absent.
"""
from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

from aekf_rts import AEKFConfig, AEKFRTS, N_X, SL

G0 = 9.80665
DEFAULT_ODR_HZ = 120.0
DT_FALLBACK = 1.0 / DEFAULT_ODR_HZ
DEFAULT_ALIGN_S = 5.0
SECOND_RTS_DEFAULT = True
SECOND_RTS_R_INFLATION = 5.0
SECOND_RTS_Q_SCALE = 1.0
REGISTRATION_STRIDE = 5
REGISTRATION_MIN_ACTIVE_DPS = 1.0
EIG_FLOOR = 1e-10

ACC_RE = re.compile(r"^(s\d+)_acc_([xyz]) \[mg\]$")
GYR_RE = re.compile(r"^(s\d+)_gyro_([xyz]) \[dps\]$")

AXES = ("X", "Y", "Z")
COLORS = {
    "fused": "black",
    "fused_rts": "tab:purple",
    "gravity": "tab:cyan",
}

QTY = {
    "a":   ("acceleration", "aE", 1e6, r"$\mu$m/s$^2$"),
    "w":   ("angular velocity", "wE", 1e3 * np.degrees(1.0), "mdps"),
    "ba":  ("accelerometer bias", "baE", 1e6, r"$\mu$m/s$^2$"),
    "bw":  ("gyroscope bias", "bwE", 1e3 * np.degrees(1.0), "mdps"),
    "g":   ("gravity vector", "gE", 1e6, r"$\mu$m/s$^2$"),
    "v":   ("velocity", "v", 1e3, "mm/s"),
    "p":   ("position", "p", 1e3, "mm"),
    "theta": ("angular position", "theta", 1e3 * np.degrees(1.0), "mdeg"),
}


# ---------------------------------------------------------------------------
# configuration helpers


def _as_vec(v, n, name):
    a = np.asarray(v, float)
    if a.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {a.shape}")
    return a.copy()


def convert_legacy_cfg(cfg0) -> AEKFConfig:
    """Accept either the new 15-state AEKFConfig or the previous 28-state cfg."""
    if isinstance(cfg0, AEKFConfig):
        return AEKFConfig(
            dt=float(cfg0.dt), G0=float(getattr(cfg0, "G0", G0)),
            tau_a=float(cfg0.tau_a), tau_w=float(cfg0.tau_w),
            P0_std=_as_vec(cfg0.P0_std, 15, "P0_std"),
            Q0_std=_as_vec(cfg0.Q0_std, 15, "Q0_std"),
            R0_std=_as_vec(cfg0.R0_std, 6, "R0_std"),
            sigma_gnorm=float(getattr(cfg0, "sigma_gnorm", 0.05)),
            enable_adaptive=bool(getattr(cfg0, "enable_adaptive", True)),
            window=int(getattr(cfg0, "window", 100)),
            b=float(getattr(cfg0, "b", 0.98)),
            min_eig_floor=float(getattr(cfg0, "min_eig_floor", EIG_FLOOR)),
            Q_min_std=None if getattr(cfg0, "Q_min_std", None) is None else _as_vec(cfg0.Q_min_std, 15, "Q_min_std"),
            R_min_std=None if getattr(cfg0, "R_min_std", None) is None else _as_vec(cfg0.R_min_std, 6, "R_min_std"),
            adapt_gyro_R=bool(getattr(cfg0, "adapt_gyro_R", False)),
            adapt_bias_process=bool(getattr(cfg0, "adapt_bias_process", False)),
            min_w_jump_std=float(getattr(cfg0, "min_w_jump_std", 0.0)),
            max_w_jump_std=getattr(cfg0, "max_w_jump_std", None),
        )

    p = np.asarray(cfg0.P0_std, float)
    q = np.asarray(cfg0.Q0_std, float)
    qmin = np.asarray(getattr(cfg0, "Q_min_std", np.zeros_like(q)), float)
    r0 = np.asarray(cfg0.R0_std, float)
    rmin = np.asarray(getattr(cfg0, "R_min_std", np.zeros(6)), float)

    if p.shape == (15,) and q.shape == (15,):
        return AEKFConfig(
            dt=float(cfg0.dt), G0=float(getattr(cfg0, "G0", G0)),
            tau_a=float(getattr(cfg0, "tau_a", 2.0)), tau_w=float(getattr(cfg0, "tau_w", 2.0)),
            P0_std=p.copy(), Q0_std=q.copy(), R0_std=r0.copy(),
            sigma_gnorm=0.05,
            enable_adaptive=bool(getattr(cfg0, "enable_adaptive", True)),
            window=int(getattr(cfg0, "window", 100)),
            b=float(getattr(cfg0, "b", 0.98)),
            min_eig_floor=float(getattr(cfg0, "min_eig_floor", EIG_FLOOR)),
            Q_min_std=qmin.copy(), R_min_std=rmin.copy(),
            adapt_gyro_R=False, adapt_bias_process=False,
        )

    if p.shape != (28,) or q.shape != (28,):
        raise ValueError("Noise configuration must contain 15-state or 28-state vectors")

    KEEP28 = np.r_[6:9, 12:15, 19:28]
    return AEKFConfig(
        dt=float(cfg0.dt), G0=float(getattr(cfg0, "G0", G0)),
        tau_a=float(getattr(cfg0, "tau_a", 2.0)), tau_w=float(getattr(cfg0, "tau_w", 2.0)),
        P0_std=p[KEEP28].copy(), Q0_std=q[KEEP28].copy(), R0_std=r0.copy(),
        sigma_gnorm=0.05,
        enable_adaptive=bool(getattr(cfg0, "enable_adaptive", True)),
        window=int(getattr(cfg0, "window", 100)),
        b=float(getattr(cfg0, "b", 0.98)),
        min_eig_floor=float(getattr(cfg0, "min_eig_floor", EIG_FLOOR)),
        Q_min_std=qmin[KEEP28].copy(), R_min_std=rmin.copy(),
        adapt_gyro_R=False, adapt_bias_process=False,
    )


R0_WINDOW_S = 5.0     # window [s] used to estimate the measurement noise (as in ekf_run.make_cfg)


def make_cfg(acc: np.ndarray, gyro: np.ndarray, dt: float) -> AEKFConfig:
    """AEKF configuration = the one of ekf_run.make_cfg (priors / process noise /
    floors), written here for the 15 states [a, w, ba, bw, g] so that ekf_run is
    NOT needed.  ekf_run uses 28 states [p v a th w q ba bw g]; the 15 kept ones
    are exactly a, w, ba, bw, g (KEEP28 = 6:9, 12:15, 19:28).  Units SI / rad/s."""
    n0 = max(2, min(len(acc), int(R0_WINDOW_S / dt)))
    R0 = np.concatenate([acc[:n0].std(0), gyro[:n0].std(0)])
    #                   a        w        ba       bw        g
    P0 = np.concatenate([[1e-3]*3, [1e-3]*3, [2e-3]*3, [2e-4]*3, [2e-3]*3])
    #  Q0: a,w = JUMP std of the piecewise white noise; ba,bw,g = random-walk std
    Q0 = np.concatenate([[2e-4]*3, [2e-4]*3, [1e-6]*3, [1e-7]*3, [1e-6]*3])
    Qmin = np.concatenate([[1e-5]*3, [1e-6]*3, [1e-8]*3, [1e-9]*3, [1e-8]*3])
    Rmin = np.concatenate([[1e-4]*3, [1e-6]*3])
    cfg = AEKFConfig(
        dt=dt, G0=G0, tau_a=2.0, tau_w=2.0,
        P0_std=P0, Q0_std=Q0, R0_std=R0, sigma_gnorm=0.05,
        enable_adaptive=True, window=100, b=0.98, min_eig_floor=EIG_FLOOR,
        Q_min_std=Qmin, R_min_std=Rmin,
        adapt_gyro_R=False, adapt_bias_process=False,
    )

    # Dynamic safeguard: deterministic angular motion must not be learned as
    # gyro measurement noise.  Also do not let the bias random walks absorb it.
    cfg.adapt_gyro_R = False
    cfg.adapt_bias_process = False

    cfg.P0_std[SL["w"]] = np.maximum(cfg.P0_std[SL["w"]], np.deg2rad(10.0))
    cfg.Q0_std[SL["w"]] = np.maximum(cfg.Q0_std[SL["w"]], np.deg2rad(0.5))
    if cfg.Q_min_std is None:
        cfg.Q_min_std = np.zeros(N_X)
    cfg.Q_min_std[SL["w"]] = np.maximum(cfg.Q_min_std[SL["w"]], np.deg2rad(0.05))
    cfg.min_w_jump_std = max(float(getattr(cfg, "min_w_jump_std", 0.0)), np.deg2rad(0.5))
    cfg.max_w_jump_std = np.deg2rad(5.0)
    return cfg


# ---------------------------------------------------------------------------
# geometry


def skew(v):
    v = np.asarray(v, float)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def normalize(v, eps=1e-15):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / max(n, eps)


def exp_so3(phi):
    return Rot.from_rotvec(np.asarray(phi, float)).as_matrix()


def gravity_alignment_R(g_body):
    """Minimum-yaw body->Earth rotation with R*g = [0,0,-1]."""
    g = normalize(g_body)
    gx, gy, gz = g
    h = max(np.hypot(gy, gz), 1e-15)
    pitch = np.arctan2(gx, h)
    roll = np.arctan2(-gy, -gz)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Ry @ Rx


def minimal_map(u, v):
    """Proper rotation C such that C @ u ~= v, with minimum angle."""
    u = normalize(u); v = normalize(v)
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    a = np.cross(u, v)
    s = np.linalg.norm(a)
    if s < 1e-12:
        if c > 0.0:
            return np.eye(3)
        ref = np.array([1., 0., 0.]) if abs(u[0]) < 0.9 else np.array([0., 1., 0.])
        axis = normalize(np.cross(u, ref))
        return exp_so3(np.pi * axis)
    axis = a / s
    return exp_so3(np.arctan2(s, c) * axis)


GW_IDX = np.r_[SL["g"].start:SL["g"].stop, SL["w"].start:SL["w"].stop]   # [g(3), w(3)]


def skew_batch(v):
    v = np.asarray(v, float)
    Z = np.zeros((len(v), 3, 3))
    Z[:, 0, 1] = -v[:, 2]; Z[:, 0, 2] = v[:, 1]
    Z[:, 1, 0] = v[:, 2];  Z[:, 1, 2] = -v[:, 0]
    Z[:, 2, 0] = -v[:, 1]; Z[:, 2, 1] = v[:, 0]
    return Z


def bd2(C):
    M = np.zeros((6, 6)); M[:3, :3] = C; M[3:, 3:] = C
    return M


def gw_blocks(x, P):
    """Joint [g, w] estimate (n,6) and full 6x6 covariance (n,6,6) of one IMU,
    including the g-w cross-covariance, taken from the smoothed 15-state."""
    return x[:, GW_IDX], P[:, GW_IDX[:, None], GW_IDX[None, :]]


def map_gw(y, P, C):
    """Rotate both 3-blocks of [g,w] by C and propagate the 6x6 covariance."""
    B = bd2(C)
    return y @ B.T, B @ P @ B.T


def _pinv_solve(H, b, rel=1e-10):
    ev, V = np.linalg.eigh(0.5 * (H + H.T))
    if ev.max() <= 0.0:
        return np.zeros(3)
    keep = ev > rel * ev.max()
    return V[:, keep] @ ((V[:, keep].T @ b) / ev[keep])


def weighted_wahba_init(T, PT, S, PS):
    """Classical weighted Wahba (Markley SVD) with scalar weights derived from
    the estimate covariances.  Used only to initialise the full-covariance
    Gauss-Newton refinement below.
        w_g = 1 / (mean tangential variance of g_t and g_s)
        w_w = 1 / (mean variance of w_t and w_s)
    so a gyro sample with |w| << sigma_w automatically gets (almost) no say,
    and no ad-hoc factor / activity threshold is needed."""
    tiny = 1e-30
    trg = np.trace(PT[:, :3, :3], axis1=1, axis2=2) + np.trace(PS[:, :3, :3], axis1=1, axis2=2)
    trw = np.trace(PT[:, 3:, 3:], axis1=1, axis2=2) + np.trace(PS[:, 3:, 3:], axis1=1, axis2=2)
    wg = 1.0 / np.maximum(trg / 2.0, tiny)
    ww = 1.0 / np.maximum(trw / 3.0, tiny)
    H = (np.einsum("n,ni,nj->ij", wg, T[:, :3], S[:, :3])
         + np.einsum("n,ni,nj->ij", ww, T[:, 3:], S[:, 3:]))
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3); D[-1, -1] = np.linalg.det(U @ Vt)
    return U @ D @ Vt


def weighted_wahba(T, PT, S, PS, C0, max_iter=15, tol=1e-10):
    """Covariance-weighted Wahba problem on SO(3), joint over gravity and
    angular rate:

        min_C  sum_k  r_k^T Sigma_k^{-1} r_k ,
        r_k     = [g_t - C g_s ; w_t - C w_s]_k
        Sigma_k = P_t,k + B(C) P_s,k B(C)^T       (full 6x6, incl. g-w cross terms)

    Solved by Gauss-Newton with C <- exp([d]x) C and Sigma re-evaluated at the
    current C (iteratively reweighted).  Returns C and the 3x3 information
    matrix H (H^-1 = covariance of the small-angle registration error)."""
    C = np.array(C0, float)
    H = np.zeros((3, 3))
    for _ in range(max_iter):
        CS, PSr = map_gw(S, PS, C)
        W = psd_inverse(PT + PSr)
        r = T - CS
        A = np.concatenate([skew_batch(CS[:, :3]), skew_batch(CS[:, 3:])], axis=1)  # r(d) = r0 + A d
        WA = W @ A
        H = np.einsum("mia,mib->ab", A, WA)
        b = np.einsum("mia,mi->a", WA, r)
        step = -_pinv_solve(H, b)
        C = exp_so3(step) @ C
        if np.linalg.norm(step) < tol:
            break
    return C, H


def registration_stats(H):
    ev, V = np.linalg.eigh(0.5 * (H + H.T))
    keep = ev > 1e-10 * max(ev.max(), 1e-300)
    cov = (V[:, keep] / ev[keep]) @ V[:, keep].T if keep.any() else np.full((3, 3), np.inf)
    smax = np.rad2deg(1.0 / np.sqrt(ev[0])) if ev[0] > 1e-10 * max(ev.max(), 1e-300) else np.inf
    return {"H": H, "cov": cov,
            "sigma_deg": np.rad2deg(np.sqrt(np.maximum(np.diag(cov), 0.0))),
            "sigma_max_deg": float(smax)}


def register_sensors(xs_by_sensor, Ps_by_sensor, sensors, stride=REGISTRATION_STRIDE,
                     active_thr_dps=REGISTRATION_MIN_ACTIVE_DPS, refine_iters=2):
    """Covariance-weighted registration of every IMU to the s1 body frame.

    1) each sensor vs s1: weighted-SVD initial guess + full-covariance
       Gauss-Newton on the joint [g, w] residual;
    2) (N>=3) Gauss-Seidel refinement: each sensor is re-registered against the
       information-fused [g, w] of ALL OTHER sensors (leave-one-out, so a sensor
       never helps define its own target and s1 noise is not the sole reference).
    s1 stays the gauge: C_s1 = I.
    `active_mask` is only a diagnostic (inverse-variance-weighted mean of |w|);
    it no longer selects samples, the covariances do the weighting."""
    ref = sensors[0]
    n = len(xs_by_sensor[ref])
    ids = np.arange(n)[::max(1, int(stride))]
    if len(ids) < 10:
        ids = np.arange(n)
    Y, P = {}, {}
    for s in sensors:
        Y[s], P[s] = gw_blocks(xs_by_sensor[s][ids], Ps_by_sensor[s][ids])

    wn = np.stack([np.linalg.norm(xs_by_sensor[s][:, SL["w"]], axis=1) for s in sensors])
    wv = np.stack([np.trace(Ps_by_sensor[s][:, SL["w"], SL["w"]], axis1=1, axis2=2) / 3.0 for s in sensors])
    wt = 1.0 / np.maximum(wv, 1e-30)
    mask = ((wt * wn).sum(0) / wt.sum(0)) > np.deg2rad(active_thr_dps)
    if mask.sum() < 10:
        mask[:] = True

    C = {ref: np.eye(3)}
    Hs = {}
    for s in sensors[1:]:
        C0 = weighted_wahba_init(Y[ref], P[ref], Y[s], P[s])
        C[s], Hs[s] = weighted_wahba(Y[ref], P[ref], Y[s], P[s], C0)

    if len(sensors) >= 3 and refine_iters > 0:
        for _ in range(int(refine_iters)):
            for s in sensors[1:]:
                ys, Pl = [], []
                for o in sensors:
                    if o == s:
                        continue
                    yo, Po = map_gw(Y[o], P[o], C[o])
                    ys.append(yo); Pl.append(Po)
                Tf, PTf = fuse_information(np.stack(ys), np.stack(Pl))
                C[s], Hs[s] = weighted_wahba(Tf, PTf, Y[s], P[s], C[s])

    stats = {ref: {"H": np.zeros((3, 3)), "cov": np.zeros((3, 3)),
                   "sigma_deg": np.zeros(3), "sigma_max_deg": 0.0}}
    for s in sensors[1:]:
        stats[s] = registration_stats(Hs[s])
        if stats[s]["sigma_max_deg"] > 1.0:
            warnings.warn(
                f"Registration for {s}: weakest direction has 1-sigma = "
                f"{stats[s]['sigma_max_deg']:.3g} deg (insufficient tilt/rotation "
                f"excitation; relative yaw may be ambiguous)."
            )
    return C, mask, stats


def consensus_body_history(xs_by_sensor, Ps_by_sensor, C, sensors, chunk=20000):
    """Covariance-weighted (information-form) consensus of [g, w] in the s1
    body frame at every sample, using each IMU's full 6x6 smoothed covariance
    mapped through its registration C_s.  Replaces the previous median.
    g is renormalised to G0 afterwards (only its direction is used)."""
    n = len(xs_by_sensor[sensors[0]])
    g = np.empty((n, 3)); w = np.empty((n, 3)); Pc = np.empty((n, 6, 6))
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        ys, Pl = [], []
        for s in sensors:
            y, Pm = gw_blocks(xs_by_sensor[s][i0:i1], Ps_by_sensor[s][i0:i1])
            y, Pm = map_gw(y, Pm, C[s])
            ys.append(y); Pl.append(Pm)
        yf, Pf = fuse_information(np.stack(ys), np.stack(Pl))
        gn = np.linalg.norm(yf[:, :3], axis=1, keepdims=True)
        g[i0:i1] = yf[:, :3] / np.maximum(gn, 1e-15) * G0
        w[i0:i1] = yf[:, 3:]
        Pc[i0:i1] = Pf
    return g, w, Pc


def integrate_orientation(w_body, R0, dt):
    n = len(w_body)
    R = np.empty((n, 3, 3)); R[0] = R0
    for k in range(n - 1):
        wm = 0.5 * (w_body[k] + w_body[k + 1])
        R[k + 1] = R[k] @ exp_so3(wm * dt)
    return R


def gravity_corrected_orientation(R_base, g_ref):
    """At each sample, use g_ref to correct tilt while preserving yaw."""
    n = len(g_ref)
    R = np.empty_like(R_base)
    gE = np.array([0.0, 0.0, -1.0])
    for k in range(n):
        g_pred_B = normalize(R_base[k].T @ gE)
        g_meas_B = normalize(g_ref[k])
        D = minimal_map(g_meas_B, g_pred_B)
        R[k] = R_base[k] @ D
    return R


# ---------------------------------------------------------------------------
# IO / data discovery


def infer_sensors(columns: Sequence[str]) -> List[str]:
    acc_s = {m.group(1) for c in columns if (m := ACC_RE.match(c))}
    gyr_s = {m.group(1) for c in columns if (m := GYR_RE.match(c))}
    sensors = sorted(acc_s & gyr_s, key=lambda x: int(x[1:]))
    if not sensors:
        raise ValueError("No complete sN accelerometer+gyro sensor groups were found")
    missing_acc = sorted(acc_s - set(sensors), key=lambda x: int(x[1:]))
    missing_gyr = sorted(gyr_s - set(sensors), key=lambda x: int(x[1:]))
    if missing_acc or missing_gyr:
        raise ValueError(f"Incomplete sensor groups: acc-only={missing_acc}, gyro-only={missing_gyr}")
    return sensors


def load_csv(path: Path, nmax: Optional[int] = None):
    df = pd.read_csv(path)
    sensors = infer_sensors(df.columns)

    # Some loggers repeat the same IMU sample across multiple rows. Keep only
    # rows where the reference sensor's accelerometer or gyroscope changed.
    # Apply this one shared row mask to every sensor to preserve synchronization.
    ref = sensors[0]
    ref_cols = (
        [f"{ref}_acc_{a} [mg]" for a in "xyz"]
        + [f"{ref}_gyro_{a} [dps]" for a in "xyz"]
    )
    X = df[ref_cols].to_numpy(float)
    keep = np.r_[True, np.any(np.diff(X, axis=0) != 0, axis=1)]
    n_before = len(df)
    df = df.iloc[keep]
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  removed {n_dropped} consecutive duplicate rows using {ref}", flush=True)

    # nmax applies to retained (unique) samples, not the unfiltered CSV rows.
    if nmax is not None:
        df = df.iloc[:nmax]
    data = {}
    for s in sensors:
        acc_cols = [f"{s}_acc_{a} [mg]" for a in "xyz"]
        gyr_cols = [f"{s}_gyro_{a} [dps]" for a in "xyz"]
        data[s] = (
            df[acc_cols].to_numpy(float) * G0,      # column says [mg], data are g (as in ekf_run)
            np.deg2rad(df[gyr_cols].to_numpy(float)),
        )
    n = min(len(v[0]) for v in data.values())
    for s in sensors:
        data[s] = data[s][0][:n], data[s][1][:n]
    return data, sensors, n


def get_dt(cli_dt: Optional[float]) -> float:
    """Return the requested sample period; default to the target 120 Hz ODR."""
    if cli_dt is not None:
        return float(cli_dt)
    return DT_FALLBACK


# ---------------------------------------------------------------------------
# Earth covariance projection


def project_earth_with_cov(xs, Ps, C, R_hist, G0=G0):
    """Project [a,w,ba,bw,g] to Earth and propagate the conditional covariance.

    The returned covariance is exact for the linear transformation induced by
    the estimated registration C and the estimated orientation R_hist.  The
    covariance of R_hist itself is not included because it depends on the
    complete preceding trajectory (cross-time attitude uncertainty). Treating
    it as an instantaneous independent covariance would be misleading and
    would also make large batch runs unnecessarily expensive.
    """
    n = len(xs)
    y15 = np.empty((n, 15))
    P15 = np.empty((n, 15, 15))
    Tcache = [None] * n
    for k in range(n):
        T = R_hist[k] @ C
        Tcache[k] = T
        a = T @ xs[k, SL["a"]]
        w = T @ xs[k, SL["w"]]
        ba = T @ xs[k, SL["ba"]]
        bw = T @ xs[k, SL["bw"]]
        g = T @ xs[k, SL["g"]]
        y15[k] = np.concatenate([a, w, ba, bw, g])

        J = np.zeros((15, N_X))
        for r0 in (0, 3, 6, 9, 12):
            J[r0:r0+3, r0:r0+3] = T
        P15[k] = J @ Ps[k] @ J.T
        P15[k] = 0.5 * (P15[k] + P15[k].T)
    return y15, P15


# ---------------------------------------------------------------------------
# covariance utilities / fusion


def psd_inverse(P, eig_floor=EIG_FLOOR):
    """Stable inverse of (batches of) approximately-PSD covariances, shape (...,d,d)."""
    P = np.asarray(P, float)
    P = 0.5 * (P + np.swapaxes(P, -1, -2))
    s = np.sqrt(np.maximum(np.diagonal(P, axis1=-2, axis2=-1), 1e-300))
    S = s[..., :, None] * s[..., None, :]
    Cm = P / S
    ev, V = np.linalg.eigh(0.5 * (Cm + np.swapaxes(Cm, -1, -2)))
    Ci = (V / np.maximum(ev, eig_floor)[..., None, :]) @ np.swapaxes(V, -1, -2)
    return Ci / S


def fuse_information(Y, P, chunk=20000):
    """Full covariance-weighted information fusion, sample by sample (batched).
    Y: (m,n,d) estimates of m sensors, P: (m,n,d,d) covariances."""
    Y = np.asarray(Y, float)
    P = np.asarray(P, float)
    m, n, d = Y.shape
    yf = np.empty((n, d)); Pf = np.empty((n, d, d))
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        L = psd_inverse(P[:, i0:i1])                       # (m,c,d,d)
        Pk = psd_inverse(L.sum(axis=0))                    # (c,d,d)
        bvec = np.einsum("mnij,mnj->ni", L, Y[:, i0:i1])
        yf[i0:i1] = np.einsum("nij,nj->ni", Pk, bvec)
        Pf[i0:i1] = 0.5 * (Pk + np.swapaxes(Pk, -1, -2))
    return yf, Pf


def fused_Q_from_individuals(results, sensors):
    """Approximate fused process covariance for the final random-walk RTS."""
    n = len(results[sensors[0]]["earth15"])
    Q = np.zeros((n, 6, 6))
    for s in sensors:
        r = results[s]
        q = r["qhist"]
        C = r["C"]
        R = r["R_hist"]
        Qa_body = q[:, SL["a"], None] * np.eye(3)[None, :, :]
        Qw_body = q[:, SL["w"], None] * np.eye(3)[None, :, :]
        for k in range(n):
            T = R[k] @ C
            Q[k, :3, :3] += T @ Qa_body[k] @ T.T
            Q[k, 3:, 3:] += T @ Qw_body[k] @ T.T
    return Q / max(len(sensors), 1)


def second_rts_random_walk(y, P, Q, r_inflation=SECOND_RTS_R_INFLATION, q_scale=SECOND_RTS_Q_SCALE):
    """6-state H=I Kalman smoother with F=I random-walk dynamics."""
    n, d = y.shape
    I = np.eye(d)
    Rm = max(float(r_inflation), 1e-12) * P
    Q = max(float(q_scale), 0.0) * Q

    xf = np.empty_like(y); Pf = np.empty_like(P)
    xp = np.empty_like(y); Pp = np.empty_like(P)
    x = y[0].copy(); Pc = P[0].copy()
    xf[0], Pf[0], xp[0], Pp[0] = x, Pc, x, Pc

    for k in range(1, n):
        xpred = x
        Ppred = Pc + Q[k]
        S = Ppred + Rm[k]
        K = Ppred @ psd_inverse(S)
        x = xpred + K @ (y[k] - xpred)
        IK = I - K
        Pc = IK @ Ppred @ IK.T + K @ Rm[k] @ K.T
        Pc = 0.5 * (Pc + Pc.T)
        xf[k], Pf[k], xp[k], Pp[k] = x, Pc, xpred, Ppred

    xs = xf.copy(); Ps = Pf.copy()
    for k in range(n - 2, -1, -1):
        Ck = Pf[k] @ psd_inverse(Pp[k + 1])
        xs[k] = xf[k] + Ck @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + Ck @ (Ps[k + 1] - Pp[k + 1]) @ Ck.T
        Ps[k] = 0.5 * (Ps[k] + Ps[k].T)
    return xs, Ps


# ---------------------------------------------------------------------------
# postprocessing / navigation


def integrate_navigation(aE, wE, dt):
    n = len(aE)
    R = np.empty((n, 3, 3)); theta = np.empty((n, 3))
    v = np.zeros((n, 3)); p = np.zeros((n, 3))
    R[0] = np.eye(3); theta[0] = 0.0
    for k in range(n - 1):
        wm = 0.5 * (wE[k] + wE[k + 1])
        R[k + 1] = exp_so3(wm * dt) @ R[k]
        theta[k + 1] = Rot.from_matrix(R[k + 1]).as_rotvec()
        v[k + 1] = v[k] + 0.5 * dt * (aE[k] + aE[k + 1])
        p[k + 1] = p[k] + 0.5 * dt * (v[k] + v[k + 1])
    return {
        "aE": aE, "wE": wE, "v": v, "p": p, "theta": theta,
        "R": R, "q_xyzw": Rot.from_matrix(R).as_quat(),
    }


def nav_from_projected(y15, dt):
    return integrate_navigation(y15[:, :3], y15[:, 3:6], dt)


# ---------------------------------------------------------------------------
# plotting


def _time(n, dt):
    return np.arange(n) * dt


def plot_quantity(path, t, sensor_results, fused_nav, fused_rts_nav, key, fused_y=None, fused_rts_y=None):
    label, field, scale, unit = QTY[key]
    fig, ax = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    for s, r in sensor_results.items():
        x = r[field] * scale
        for j in range(3):
            ax[j].plot(t, x[:, j], lw=0.7, alpha=0.7, label=s if j == 0 else None)
    if fused_nav is not None:
        xf = fused_nav[field] if field in fused_nav else fused_y
        if xf is not None:
            for j in range(3):
                ax[j].plot(t, xf[:, j] * scale, lw=2.0, color=COLORS["fused"], label="weighted fusion" if j == 0 else None)
    if fused_rts_nav is not None:
        xr = fused_rts_nav[field] if field in fused_rts_nav else fused_rts_y
        if xr is not None:
            for j in range(3):
                ax[j].plot(t, xr[:, j] * scale, lw=2.0, ls="--", color=COLORS["fused_rts"], label="weighted fusion + second RTS" if j == 0 else None)
    for j in range(3):
        ax[j].set_ylabel(f"{AXES[j]} [{unit}]")
        ax[j].grid(alpha=0.25)
    ax[0].set_title(label)
    ax[0].legend(fontsize=8)
    ax[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_biases(path, t, sensor_results, key):
    label, field, scale, unit = QTY[key]
    fig, ax = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    for s, r in sensor_results.items():
        x = r[field] * scale
        for j in range(3):
            ax[j].plot(t, x[:, j], lw=0.75, alpha=0.75, label=s if j == 0 else None)
    for j in range(3):
        ax[j].set_ylabel(f"{AXES[j]} [{unit}]")
        ax[j].grid(alpha=0.25)
    ax[0].set_title(label + " (individual IMUs)")
    ax[0].legend(fontsize=8)
    ax[-1].set_xlabel("time [s]")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_fusion_comparison(path, t, fused_nav, fused_rts_nav, key):
    label, field, scale, unit = QTY[key]
    fig, ax = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    for j in range(3):
        ax[j].plot(t, fused_nav[field][:, j] * scale, color=COLORS["fused"], lw=1.7, label="weighted fusion")
        if fused_rts_nav is not None:
            ax[j].plot(t, fused_rts_nav[field][:, j] * scale, color=COLORS["fused_rts"], ls="--", lw=1.8, label="weighted fusion + second RTS")
        ax[j].set_ylabel(f"{AXES[j]} [{unit}]"); ax[j].grid(alpha=0.25)
    ax[0].set_title(label + " — fusion comparison")
    ax[0].legend(fontsize=8); ax[-1].set_xlabel("time [s]")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

def plot_norm(path, t, sensor_results, fused_nav, fused_rts_nav, key):
    """Plot the modulus (Euclidean norm) of a 3-vector quantity.

    Useful for position and angular position, where the norm is the natural
    scalar to compare per-IMU vs covariance-weighted fusion.
    """
    label, field, scale, unit = QTY[key]
    fig, ax = plt.subplots(figsize=(15, 5))

    # Individual IMUs
    for s, r in sensor_results.items():
        x = r[field] * scale
        ax.plot(t, np.linalg.norm(x, axis=1), lw=0.8, alpha=0.75, label=s)

    # Covariance-weighted fusion
    if fused_nav is not None and field in fused_nav:
        xf = fused_nav[field] * scale
        ax.plot(t, np.linalg.norm(xf, axis=1), lw=2.2,
                color=COLORS["fused"], label="weighted fusion")

    # Fusion + second RTS
    if fused_rts_nav is not None and field in fused_rts_nav:
        xr = fused_rts_nav[field] * scale
        ax.plot(t, np.linalg.norm(xr, axis=1), lw=2.2, ls="--",
                color=COLORS["fused_rts"], label="weighted fusion + second RTS")

    ax.set_ylabel(f"|{field}| [{unit}]")
    ax.set_xlabel("time [s]")
    ax.set_title(f"{label} — modulus |{field}|")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

def constrain_gnorm(g, g0=G0):
    """Gravity-vector constraint ||g|| = G0: same direction, modulus fixed to G0.
    Applied to the final g outputs only (nothing upstream is modified)."""
    g = np.asarray(g, float)
    return g * (g0 / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12))


def plot_gnorm(path, t, sensor_results, fused_g=None):
    """History of the modulus of the gravity vector |g| (per IMU, Earth-projected)."""
    fig, ax = plt.subplots(figsize=(15, 5))
    for s, r in sensor_results.items():
        ax.plot(t, np.linalg.norm(r["gE"], axis=1), lw=0.9, alpha=0.8, label=s)
    if fused_g is not None:
        ax.plot(t, np.linalg.norm(fused_g, axis=1), lw=2.0, ls="--", color=COLORS["fused"],
                label="consensus")
    ax.axhline(G0, color="r", ls=":", lw=1.2, label=f"G0 = {G0:.5f}")
    ax.set_ylabel(r"$|g|$ [m/s$^2$]"); ax.set_xlabel("time [s]")
    ax.set_title("gravity vector — modulus |g|")
    ax.grid(alpha=0.25); ax.legend(fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


GIF_COLORS = ("tab:orange", "tab:blue", "tab:green", "tab:red", "tab:brown", "tab:pink", "tab:olive", "tab:cyan")


def make_dynamics_gifs(file_out, sensors, results, nav_f, nav_rts, dt):
    """One GIF per IMU: IMU estimate vs final fused estimate, Earth frame.
    Attitude R_abs(k) = R_rel(k) @ R_abs(0) with R_rel the integrated Earth angular
    rate (same as `theta`) and R_abs(0) = R_hist[0] @ C (sensor -> Earth at t0)."""
    from imu_gifs import make_imu_dynamics_gif
    nav_fin = nav_rts if nav_rts is not None else nav_f
    fused_label = "fused + 2nd RTS" if nav_rts is not None else "fused"
    fused_color = COLORS["fused_rts"] if nav_rts is not None else COLORS["fused"]
    gif_dir = file_out / "gifs"; gif_dir.mkdir(exist_ok=True)
    for i, s in enumerate(sensors):
        R0s = results[s]["R_hist"][0] @ results[s]["C"]
        nav = results[s]["nav"]
        imu = dict(R=nav["R"] @ R0s, p=nav["p"], w=nav["wE"])
        fus = dict(R=nav_fin["R"] @ R0s, p=nav_fin["p"], w=nav_fin["wE"])
        out = gif_dir / f"dynamics_{s}_vs_fused.gif"
        make_imu_dynamics_gif(out, s, dt, imu, fus, imu_color=GIF_COLORS[i % len(GIF_COLORS)],
                              fused_color=fused_color, fused_label=fused_label)
        print(f"  saved {out.name}", flush=True)


# ---------------------------------------------------------------------------
# main per-file pipeline


def process_file(path: Path, out_root: Path, dt: float, align_s: float,
                 second_rts: bool, r_inflation: float, q_scale: float,
                 registration_stride: int, active_thr_dps: float,
                 nmax: Optional[int], registration_refine_iters: int = 2):
    data, sensors, n = load_csv(path, nmax=nmax)
    file_out = out_root / path.stem
    file_out.mkdir(parents=True, exist_ok=True)

    xs = {}; Ps = {}; qhist = {}; raw_acc = {}; raw_gyro = {}; cfgs = {}
    print(f"\n=== {path.name}: {n} samples, {len(sensors)} IMUs ({', '.join(sensors)}) ===", flush=True)

    for s in sensors:
        acc, gyro = data[s]
        cfg = make_cfg(acc, gyro, dt)
        na = max(1, min(n, int(round(align_s / dt))))
        x0 = AEKFRTS.stationary_initial_state(acc[:na].mean(0), gyro[:na].mean(0), cfg.G0)
        ekf = AEKFRTS(cfg, x0)
        out = ekf.run(acc, gyro, do_smooth=True)
        xs[s] = out["xs"]
        Ps[s] = out["Ps"]
        qhist[s] = np.asarray(out["qhist"])
        raw_acc[s] = acc
        raw_gyro[s] = gyro
        cfgs[s] = cfg
        print(f"  finished AEKF+RTS {s}", flush=True)

    C, active_mask, reg_stats = register_sensors(
        xs, Ps, sensors, stride=registration_stride, active_thr_dps=active_thr_dps,
        refine_iters=registration_refine_iters,
    )
    for s in sensors[1:]:
        print(f"  registration {s}->{sensors[0]}: 1-sigma [deg] = "
              f"{np.array2string(reg_stats[s]['sigma_deg'], precision=3)}", flush=True)
    g_cons, w_cons, P_cons = consensus_body_history(xs, Ps, C, sensors)
    R0 = gravity_alignment_R(g_cons[0])  # yaw(t=0)=0 convention
    R_base = integrate_orientation(w_cons, R0, dt)

    results = {}
    for s in sensors:
        g_ref = np.einsum("ij,nj->ni", C[s], xs[s][:, SL["g"]])
        R_hist = gravity_corrected_orientation(R_base, g_ref)
        Y15, P15 = project_earth_with_cov(xs[s], Ps[s], C[s], R_hist, G0)
        nav = nav_from_projected(Y15, dt)
        results[s] = {
            "xs": xs[s], "Ps": Ps[s], "qhist": qhist[s], "C": C[s],
            "R_hist": R_hist, "earth15": Y15, "earthP15": P15,
            "aE": Y15[:, 0:3], "wE": Y15[:, 3:6],
            "baE": Y15[:, 6:9], "bwE": Y15[:, 9:12], "gE": constrain_gnorm(Y15[:, 12:15]),
            "aE_P": P15[:, 0:3, 0:3], "wE_P": P15[:, 3:6, 3:6],
            "nav": nav,
            "raw_acc": raw_acc[s], "raw_gyro": raw_gyro[s],
        }
        print(f"  projected Earth frame {s}", flush=True)

    Y = np.stack([results[s]["earth15"][:, 0:6] for s in sensors])
    P = np.stack([results[s]["earthP15"][:, 0:6, 0:6] for s in sensors])
    y_f, p_f = fuse_information(Y, P)
    nav_f = integrate_navigation(y_f[:, :3], y_f[:, 3:6], dt)

    y_rts = p_rts = nav_rts = None
    if second_rts:
        Q = fused_Q_from_individuals(results, sensors)
        y_rts, p_rts = second_rts_random_walk(
            y_f, p_f, Q, r_inflation=r_inflation, q_scale=q_scale
        )
        nav_rts = integrate_navigation(y_rts[:, :3], y_rts[:, 3:6], dt)

    t = _time(n, dt)
    sensor_plot = {s: {
        "aE": results[s]["aE"], "wE": results[s]["wE"],
        "baE": results[s]["baE"], "bwE": results[s]["bwE"], "gE": results[s]["gE"],
        "v": results[s]["nav"]["v"], "p": results[s]["nav"]["p"], "theta": results[s]["nav"]["theta"],
    } for s in sensors}

    # Plots: all requested quantities in Earth frame. Biases are individual-only.
    for q in ("a", "w", "v", "p", "theta"):
        plot_quantity(file_out / f"{q}.png", t, sensor_plot, nav_f, nav_rts, q,
                      fused_y=y_f, fused_rts_y=y_rts)
        plot_fusion_comparison(file_out / f"{q}_fusion_compare.png", t, nav_f, nav_rts, q)
        # Modulus plots for position and angular position (per-IMU vs fused).
    for q in ("p", "theta"):
        plot_norm(file_out / f"{q}_norm.png", t, sensor_plot,
                  nav_f, nav_rts, q)
        plot_norm(file_out / f"{q}_norm_fusion_compare.png", t, {}, nav_f, nav_rts, q)
    for q in ("ba", "bw", "g"):
        plot_biases(file_out / f"{q}.png", t, sensor_plot, q)
    plot_gnorm(file_out / "g_norm.png", t, sensor_plot, constrain_gnorm(g_cons))

    # Additional diagnostics directly at the filter output, before registration.
    diag_rows = []
    for s in sensors:
        w_raw = raw_gyro[s]
        w_est = xs[s][:, SL["w"]]
        raw_rms = np.sqrt(np.mean(np.sum(w_raw * w_raw, axis=1)))
        est_rms = np.sqrt(np.mean(np.sum(w_est * w_est, axis=1)))
        diag_rows.append({
            "file": path.name, "sensor": s,
            "raw_gyro_rms_dps": np.rad2deg(raw_rms),
            "aekf_rts_w_rms_dps": np.rad2deg(est_rms),
            "aekf_rts_w_max_dps": np.rad2deg(np.max(np.linalg.norm(w_est, axis=1))),
            "registration_sigma_max_deg": reg_stats[s]["sigma_max_deg"],
        })
    pd.DataFrame(diag_rows).to_csv(file_out / "filter_diagnostic.csv", index=False)

    # Save compact output for downstream analysis.
    save_dict = {"t": t, "sensors": np.array(sensors, dtype=object), "R0": R0,
                 "g_consensus_B": g_cons, "w_consensus_B": w_cons, "P_consensus_gw_B": P_cons,
                 "active_mask": active_mask, "y_fused": y_f, "P_fused": p_f}
    if y_rts is not None:
        save_dict.update({"y_fused_rts": y_rts, "P_fused_rts": p_rts})
    for s in sensors:
        r = results[s]
        save_dict.update({
            f"{s}_xf": np.asarray(r["xs"]),
            f"{s}_Pf": np.asarray(r["Ps"]),
            f"{s}_C": r["C"], f"{s}_C_cov_rad2": reg_stats[s]["cov"], f"{s}_R_hist": r["R_hist"],
            f"{s}_earth15": r["earth15"], f"{s}_earthP15": r["earthP15"],
            f"{s}_aE": r["aE"], f"{s}_wE": r["wE"],
            f"{s}_baE": r["baE"], f"{s}_bwE": r["bwE"], f"{s}_gE": r["gE"],
            f"{s}_v": r["nav"]["v"], f"{s}_p": r["nav"]["p"], f"{s}_theta": r["nav"]["theta"],
        })
    save_dict.update({
        "a_fused": nav_f["aE"], "w_fused": nav_f["wE"], "v_fused": nav_f["v"],
        "p_fused": nav_f["p"], "theta_fused": nav_f["theta"],
    })
    if nav_rts is not None:
        save_dict.update({
            "a_fused_rts": nav_rts["aE"], "w_fused_rts": nav_rts["wE"],
            "v_fused_rts": nav_rts["v"], "p_fused_rts": nav_rts["p"], "theta_fused_rts": nav_rts["theta"],
        })
    np.savez_compressed(file_out / "results.npz", **save_dict)

    make_dynamics_gifs(file_out, sensors, results, nav_f, nav_rts, dt)

    # One-line numerical summary useful for batch runs.
    summary = {
        "file": path.name, "n_samples": n, "n_imus": len(sensors), "dt_s": dt,
        "duration_s": (n - 1) * dt,
        "fused_a_rms_mps2": float(np.sqrt(np.mean(np.sum(nav_f["aE"]**2, axis=1)))),
        "fused_w_rms_dps": float(np.rad2deg(np.sqrt(np.mean(np.sum(nav_f["wE"]**2, axis=1))))),
        "fused_v_end_mmps": float(1e3 * np.linalg.norm(nav_f["v"][-1])),
        "fused_p_end_mm": float(1e3 * np.linalg.norm(nav_f["p"][-1])),
        "fused_theta_end_mdeg": float(np.rad2deg(1e3 * np.linalg.norm(nav_f["theta"][-1]))),
        "second_rts": bool(second_rts),
    }
    if nav_rts is not None:
        summary.update({
            "fused_rts_a_rms_mps2": float(np.sqrt(np.mean(np.sum(nav_rts["aE"]**2, axis=1)))),
            "fused_rts_w_rms_dps": float(np.rad2deg(np.sqrt(np.mean(np.sum(nav_rts["wE"]**2, axis=1))))),
            "fused_rts_v_end_mmps": float(1e3 * np.linalg.norm(nav_rts["v"][-1])),
            "fused_rts_p_end_mm": float(1e3 * np.linalg.norm(nav_rts["p"][-1])),
            "fused_rts_theta_end_mdeg": float(np.rad2deg(1e3 * np.linalg.norm(nav_rts["theta"][-1]))),
        })
    pd.DataFrame([summary]).to_csv(file_out / "summary.csv", index=False)
    print(f"  saved {file_out}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# CLI


def main():
    ap = argparse.ArgumentParser(description="General arbitrary-N IMU dynamic multisensor pipeline")
    ap.add_argument("--data-dir", default=None, help="Input folder; default is DATA next to this script")
    ap.add_argument("--output-dir", default=None, help="Output folder; default outputs_general_dynamic next to this script")
    ap.add_argument("--pattern", default="*.csv", help="CSV glob pattern inside DATA")
    ap.add_argument("--files", nargs="*", default=None, help="Optional exact filenames relative to DATA")
    ap.add_argument("--dt", type=float, default=None, help="Sample period [s]; default 1/120 for 120 Hz ODR")
    ap.add_argument("--align-s", type=float, default=DEFAULT_ALIGN_S, help="Initial stationary interval [s]")
    ap.add_argument("--nmax", type=int, default=None)
    ap.add_argument("--no-second-rts", action="store_true", help="Disable the post-fusion 6-state RTS")
    ap.add_argument("--r-inflation", type=float, default=SECOND_RTS_R_INFLATION)
    ap.add_argument("--q-scale", type=float, default=SECOND_RTS_Q_SCALE)
    ap.add_argument("--registration-stride", type=int, default=REGISTRATION_STRIDE)
    ap.add_argument("--active-threshold-dps", type=float, default=REGISTRATION_MIN_ACTIVE_DPS,
                    help="diagnostic activity mask only; registration is covariance-weighted")
    ap.add_argument("--registration-refine-iters", type=int, default=2,
                    help="leave-one-out Gauss-Seidel refinement sweeps (N>=3)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else root / "DATA"
    out_dir = Path(args.output_dir) if args.output_dir else root / "outputs_general_dynamic"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA folder does not exist: {data_dir}")

    if args.files:
        paths = [data_dir / f for f in args.files]
    else:
        paths = sorted(data_dir.glob(args.pattern))
    paths = [p for p in paths if p.suffix.lower() == ".csv"]
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir} matching {args.pattern!r}")

    dt = get_dt(args.dt)
    summaries = []
    for path in paths:
        summaries.append(process_file(
            path=path,
            out_root=out_dir,
            dt=dt,
            align_s=args.align_s,
            second_rts=not args.no_second_rts,
            r_inflation=args.r_inflation,
            q_scale=args.q_scale,
            registration_stride=args.registration_stride,
            active_thr_dps=args.active_threshold_dps,
            nmax=args.nmax,
            registration_refine_iters=args.registration_refine_iters,
        ))
    pd.DataFrame(summaries).to_csv(out_dir / "batch_summary.csv", index=False)
    print(f"\nProcessed {len(summaries)} CSV file(s). Batch summary: {out_dir / 'batch_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()