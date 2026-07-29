"""
render.py — rot139 features -> stick-figure GIF + static frame grid.

Crystallized from the research renderer (
animate_motion.py) onto first-class package imports: the SMPL body model and
the feature<->SMPL converters live in modalities.motion.data.converters.

A motion = features [T,139] -> SMPL forward kinematics -> global joint
positions [T,22,3] -> an animated 3D stick figure. Axis limits are handled so
you can SEE root translation (a known rot139 weakness) instead of per-frame
autoscaling hiding it: follow=True tracks the root (body stays large, sliding
feet still betray drift); follow=False fixes one cube over the whole clip.
"""

import numpy as np

from modalities.motion.data.converters import smpl_body as B
from modalities.motion.data.converters import smpl_to_rot139 as conv

SMPL_BONES = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9),
              (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17),
              (16, 18), (17, 19), (18, 20), (19, 21)]


def features_to_gp(feats):
    """features [T,139] -> global joint positions [T,22,3] via SMPL FK."""
    J, parents = B.load_body_model("neutral")
    R, tr = conv.features_to_smpl(np.asarray(feats, dtype=np.float64), np.zeros(3))
    gp, _ = B.smpl_fk(R, tr, J, parents)
    return gp


def gp_to_gif(gp, path, fps=20, title="", max_frames=160, follow=True, span=1.3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    T = len(gp)
    if T > max_frames:                              # subsample long clips for a light gif
        idx = np.linspace(0, T - 1, max_frames).astype(int)
        gp = gp[idx]
    up = B.UP_AXIS
    h0, h1 = B.HORIZ_AXES
    mn, mx = gp.min(axis=(0, 1)), gp.max(axis=(0, 1))
    ctr = (mn + mx) / 2
    fixed_span = max((mx - mn).max(), 1.0) / 2 * 1.1
    z_top = (mx[up] if follow else ctr[up] + fixed_span)

    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111, projection="3d")

    def draw(t):
        ax.cla()
        P = gp[t]
        for a, b in SMPL_BONES:
            ax.plot([P[a, h0], P[b, h0]], [P[a, h1], P[b, h1]], [P[a, up], P[b, up]],
                    "-o", ms=2, lw=2, color="steelblue")
        if follow:                                  # centre on the root horizontally
            cx, cy = P[0, h0], P[0, h1]
            ax.set_xlim(cx - span, cx + span); ax.set_ylim(cy - span, cy + span)
            ax.set_zlim(min(0, mn[up]), max(z_top, mn[up] + 2 * span))
        else:
            ax.set_xlim(ctr[h0] - fixed_span, ctr[h0] + fixed_span)
            ax.set_ylim(ctr[h1] - fixed_span, ctr[h1] + fixed_span)
            ax.set_zlim(min(ctr[up] - fixed_span, 0), ctr[up] + fixed_span)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(f"{title}\nframe {t+1}/{len(gp)}", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("up")

    anim = FuncAnimation(fig, draw, frames=len(gp), interval=1000 / fps)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close()
    return path


def features_to_gif(feats, path, fps=20, title="", **kw):
    return gp_to_gif(features_to_gp(feats), path, fps=fps, title=title, **kw)


def save_grid(samples, path, n_frames=6):
    """[(label, features)] -> one static PNG: a row of frames per sample."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid = [(lab, f) for lab, f in samples if f is not None]
    if not valid:
        return None
    up, h = B.UP_AXIS, B.HORIZ_AXES[0]
    J, parents = B.load_body_model("neutral")
    bones = [(int(parents[j]), j) for j in range(1, B.N_BODY_JOINTS)]
    fig, axes = plt.subplots(len(valid), n_frames,
                             figsize=(2.1 * n_frames, 2.5 * len(valid)))
    axes = np.atleast_2d(axes)
    for r, (lab, feats) in enumerate(valid):
        gp = features_to_gp(feats)
        for c, t in enumerate(np.linspace(0, len(gp) - 1, n_frames).astype(int)):
            ax = axes[r][c]
            for a, b in bones:
                ax.plot([gp[t, a, h], gp[t, b, h]], [gp[t, a, up], gp[t, b, up]],
                        "-o", ms=1.5, lw=1, color="steelblue")
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(lab[:22], fontsize=8)
    fig.suptitle("generated motion")
    fig.tight_layout()
    fig.savefig(path, dpi=95)
    plt.close()
    return path
