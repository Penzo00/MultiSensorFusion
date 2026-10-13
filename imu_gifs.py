"""imu_gifs.py
One GIF per IMU: that IMU's estimate vs the final fused estimate, both seen from the
inertial (Earth) frame, using the same Earth-frame data that produce the PNG plots.

Left   : 3D trajectory p(t) in Earth XYZ [mm] (trail + current position), with the
         body (plate + its x/y/z axes) drawn at the current position with its
         current Earth-frame attitude.
Right  : top    - the rotation alone: both bodies at a common pivot, Earth XYZ axes
         middle - position x/y/z [mm] vs time
         bottom - cumulative rotation  int w_E dt  [deg] vs time
Solid/coloured = single IMU, dashed/purple-or-black = fused.

Track dict (Earth frame):  "R" (n,3,3) body->Earth attitude, "p" (n,3) position [m],
                           "w" (n,3) angular rate [rad/s].
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

AX_COL = ("tab:red", "tab:green", "tab:blue")      # body x, y, z
FPS = 20
MAX_FRAMES = 220
DPI = 70


def frame_indices(w, p, dt, max_frames=MAX_FRAMES, rest=0.15):
    """Frames equally spaced in (angle travelled + distance travelled + a small time
    term): smooth motion while moving, fast-forward while still."""
    wn = np.linalg.norm(w, axis=1)
    sp = np.linalg.norm(np.gradient(p, dt, axis=0), axis=1)
    r = wn / max(np.percentile(wn, 95), 1e-12) + sp / max(np.percentile(sp, 95), 1e-12) + rest
    s = np.concatenate([[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * dt)])
    idx = np.unique(np.clip(np.searchsorted(s, np.linspace(0.0, s[-1], max_frames)), 0, len(p) - 1))
    if idx[-1] != len(p) - 1:
        idx = np.append(idx, len(p) - 1)
    return idx


def _plate(R, center, half):
    sx, sy, sz = half
    v = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    v = center + v @ R.T
    return [v[list(q)] for q in [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
                                  (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]]


def geodesic_deg(Ra, Rb):
    c = 0.5 * (np.einsum("nij,nij->n", Ra, Rb) - 1.0)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def make_imu_dynamics_gif(path, sensor, dt, imu, fused, imu_color="tab:orange",
                          fused_color="black", fused_label="fused"):
    n = len(imu["R"])
    t = np.arange(n) * dt
    T = {"imu": imu, "fused": fused}
    col = {"imu": imu_color, "fused": fused_color}
    lab = {"imu": sensor, "fused": fused_label}
    R = {k: np.asarray(v["R"], float) for k, v in T.items()}
    P = {k: np.asarray(v["p"], float) * 1e3 for k, v in T.items()}                 # mm
    TH = {k: np.degrees(np.cumsum(np.asarray(v["w"], float), axis=0) * dt) for k, v in T.items()}
    diff = geodesic_deg(R["imu"], R["fused"])
    idx = frame_indices(np.asarray(fused["w"], float), np.asarray(fused["p"], float), dt)

    # cubic axis box around both trajectories
    allp = np.concatenate([P["imu"], P["fused"]])
    lo, hi = allp.min(0), allp.max(0)
    ctr = 0.5 * (lo + hi)
    half = max(0.6 * float(np.max(hi - lo)), 1.0)                                  # >= 1 mm
    body = (0.10 * half, 0.06 * half, 0.012 * half)                                # plate half-sizes
    Lax = 0.16 * half                                                              # body-axis length

    fig = plt.figure(figsize=(14.5, 8.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1.0], height_ratios=[1.25, 0.8, 0.8],
                          hspace=0.55, wspace=0.10, left=0.01, right=0.985, top=0.93, bottom=0.06)
    ax = fig.add_subplot(gs[:, 0], projection="3d")
    axr = fig.add_subplot(gs[0, 1], projection="3d")
    axp = fig.add_subplot(gs[1, 1])
    axt = fig.add_subplot(gs[2, 1])
    title = fig.suptitle("", fontsize=12, weight="bold", family="monospace", x=0.01, ha="left")

    # ---- trajectory view
    ax.set_xlim(ctr[0] - half, ctr[0] + half); ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half); ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm] (up)")
    ax.view_init(elev=22, azim=-58)
    ax.plot([P["imu"][0, 0]], [P["imu"][0, 1]], [P["imu"][0, 2]], marker="*", ms=12,
            color="gold", mec="k", ls="none", label="start")
    art = {}
    for k in T:
        trail, = ax.plot([], [], [], color=col[k], lw=1.8 if k == "imu" else 1.4,
                         ls="-" if k == "imu" else "--", label=lab[k])
        head, = ax.plot([], [], [], marker="o", ms=5, color=col[k], ls="none")
        faces = Poly3DCollection(_plate(np.eye(3), np.zeros(3), body), facecolors=col[k],
                                 edgecolors=col[k] if k == "imu" else "k",
                                 alpha=0.35 if k == "imu" else 0.15, linewidths=1.0)
        ax.add_collection3d(faces)
        axes = [ax.plot([], [], [], color=AX_COL[j], lw=2.6 if k == "imu" else 1.4,
                        ls="-" if k == "imu" else "--")[0] for j in range(3)]
        art[k] = dict(trail=trail, head=head, faces=faces, axes=axes)
    ax.legend(loc="upper left", fontsize=8)

    # ---- rotation view (common pivot)
    axr.set_xlim(-1, 1); axr.set_ylim(-1, 1); axr.set_zlim(-1, 1); axr.set_box_aspect((1, 1, 1))
    axr.view_init(elev=20, azim=-58)
    axr.set_xticklabels([]); axr.set_yticklabels([]); axr.set_zticklabels([])
    axr.set_title("rotation in Earth frame (common pivot)   body axes x/y/z = red/green/blue",
                  fontsize=8.5, pad=-4)
    for e, txt in zip(np.eye(3), "XYZ"):
        axr.plot([0, 1.0 * e[0]], [0, 1.0 * e[1]], [0, 1.0 * e[2]], color="0.4", lw=1.0, ls=":")
        axr.text(*(1.1 * e), txt, color="0.25", fontsize=9)
    rart = {}
    for k in T:
        faces = Poly3DCollection(_plate(np.eye(3), np.zeros(3), (0.5, 0.32, 0.06)),
                                 facecolors=col[k], edgecolors=col[k] if k == "imu" else "k",
                                 alpha=0.30 if k == "imu" else 0.12, linewidths=1.0)
        axr.add_collection3d(faces)
        axes = [axr.plot([], [], [], color=AX_COL[j], lw=3.0 if k == "imu" else 1.5,
                         ls="-" if k == "imu" else "--")[0] for j in range(3)]
        rart[k] = dict(faces=faces, axes=axes)

    # ---- time histories
    prog = {}
    def style(a, yl, ttl):
        a.set_ylabel(yl, fontsize=8); a.set_title(ttl, fontsize=8.5, loc="left", pad=2)
        a.grid(alpha=0.25); a.tick_params(labelsize=7); a.set_xlim(t[0], t[-1])
    style(axp, "mm", "position p  (x/y/z = red/green/blue, solid = IMU, dashed = fused)")
    style(axt, "deg", "cumulative rotation  ∫ω_E dt  (x/y/z = red/green/blue)")
    for a, D, name in ((axp, P, "p"), (axt, TH, "t")):
        for k in T:
            for j in range(3):
                a.plot(t, D[k][:, j], color=AX_COL[j], lw=2.0 if k == "imu" else 1.2,
                       ls="-" if k == "imu" else "--", alpha=0.25)
                prog[(name, k, j)] = a.plot([], [], color=AX_COL[j], lw=2.0 if k == "imu" else 1.2,
                                            ls="-" if k == "imu" else "--")[0]
    axt.set_xlabel("time [s]", fontsize=8)
    cursors = [axp.axvline(0, color="k", lw=0.9), axt.axvline(0, color="k", lw=0.9)]

    def setline(ln, a, b):
        ln.set_data([a[0], b[0]], [a[1], b[1]]); ln.set_3d_properties([a[2], b[2]])

    def update(fi):
        kk = int(idx[fi])
        sp = (t[idx[fi + 1]] - t[kk]) * FPS if fi + 1 < len(idx) else 0.0
        title.set_text(f"{sensor} vs {fused_label}   t = {t[kk]:6.2f} s   playback x{sp:.1f}")
        sl = slice(0, kk + 1, max(1, kk // 1500 + 1))
        for k in T:
            Rk = R[k][kk]; pk = P[k][kk]; a = art[k]
            seg = P[k][sl]
            a["trail"].set_data(seg[:, 0], seg[:, 1]); a["trail"].set_3d_properties(seg[:, 2])
            a["head"].set_data([pk[0]], [pk[1]]); a["head"].set_3d_properties([pk[2]])
            a["faces"].set_verts(_plate(Rk, pk, body))
            r = rart[k]
            r["faces"].set_verts(_plate(Rk, np.zeros(3), (0.5, 0.32, 0.06)))
            for j in range(3):
                setline(a["axes"][j], pk, pk + Lax * Rk[:, j])
                setline(r["axes"][j], np.zeros(3), 0.85 * Rk[:, j])
            for j in range(3):
                prog[("p", k, j)].set_data(t[sl], P[k][sl, j])
                prog[("t", k, j)].set_data(t[sl], TH[k][sl, j])
        for c in cursors:
            c.set_xdata([t[kk], t[kk]])
        return []

    ani = FuncAnimation(fig, update, frames=len(idx), blit=False)
    ani.save(str(path), writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
