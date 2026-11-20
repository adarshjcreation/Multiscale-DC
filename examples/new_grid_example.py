"""Example: downward continuation of a new gravity grid with the pretrained multiscale networks.

A 3-D density model that is not part of any training or test set (64 x 96 cells of 50 m, six bodies with
positive and negative density contrasts) is forward-modelled with the exact prism formula:
the observations 100 m above the ground and, for comparison, the true field on the ground. The observations
are then continued 100 m downward with ``multiscale_dc.downward_continue`` and compared with the truth.

    python examples/new_grid_example.py          # writes examples/data/*.csv and docs/new_grid_example.png
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from multiscale_dc import downward_continue, save_grid  # noqa: E402
from multiscale_dc.forward import grid_gravity  # noqa: E402

CELL = 50.0      # m, cell size = grid spacing
HEIGHT = 100.0   # m, observation height above the ground (= 2 grid spacings)
NZ, NY, NX = 16, 64, 96
PROFILE_ROW = 16


def density_model():
    """Six bodies (density contrast in kg/m^3); layer k spans depths 50 k to 50 (k + 1) m."""
    m = np.zeros((NZ, NY, NX))
    bodies = []

    def add(name, rho, rows, cols, layers):
        m[layers[0]:layers[1], rows[0]:rows[1], cols[0]:cols[1]] = rho
        bodies.append((name, rho, rows, cols, layers))

    add("shallow ore body", 1000, (14, 19), (12, 17), (0, 3))      # 0-150 m deep
    add("twin body 1", 800, (15, 18), (32, 35), (1, 3))            # 50-150 m, 150 m apart from its twin
    add("twin body 2", 800, (15, 18), (38, 41), (1, 3))
    add("deep block", 400, (6, 26), (64, 86), (8, 14))             # 400-700 m
    add("low-density basin", -600, (34, 48), (56, 80), (1, 5))      # 50-250 m
    for k in range(1, 9):                                          # dipping dyke, 50-450 m, dips to the north
        m[k, 50 + k // 2:52 + k // 2, 14:46] = 500
    bodies.append(("dipping dyke", 500, (50, 56), (14, 46), (1, 9)))
    return m, bodies


def psnr(truth, est, border=4):
    t, e = truth[border:-border, border:-border], est[border:-border, border:-border]
    return 10 * np.log10(np.ptp(t) ** 2 / np.mean((t - e) ** 2))


def figure(m, bodies, obs, est, truth, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    ext = [0, NX * CELL / 1000, 0, NY * CELL / 1000]  # km
    x_km = (np.arange(NX) + 0.5) * CELL / 1000
    y_prof = (PROFILE_ROW + 0.5) * CELL / 1000
    norm = TwoSlopeNorm(vmin=truth.min(), vcenter=0, vmax=truth.max())
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.4), constrained_layout=True)

    ax = axes[0, 0]
    top = np.zeros((NY, NX))
    for k in range(NZ - 1, -1, -1):  # density of the shallowest non-zero cell
        top = np.where(m[k] != 0, m[k], top)
    im = ax.imshow(top, origin="lower", extent=ext, cmap="RdBu_r", vmin=-1000, vmax=1000)
    for name, rho, rows, cols, layers in bodies:
        if name == "twin body 2":
            continue
        label = "twin bodies" if name == "twin body 1" else name
        depth = f"{layers[0] * CELL:.0f}-{layers[1] * CELL:.0f} m deep"
        xc = (cols[0] + (cols[1] if name != "twin body 1" else 41)) / 2 * CELL / 1000
        if name in ("shallow ore body", "dipping dyke"):  # labels below: clear of the twin bodies and the title
            ax.annotate(f"{label}\n{depth}", (xc, rows[0] * CELL / 1000 - 0.08), ha="center", va="top", fontsize=7.5)
        else:
            ax.annotate(f"{label}\n{depth}", (xc, rows[1] * CELL / 1000 + 0.08), ha="center", va="bottom", fontsize=7.5)
    ax.set_title("(a) New density model (plan view)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="density contrast (kg/m³)")

    maps = [(axes[0, 1], obs, f"(b) Input: observed at {HEIGHT:.0f} m height\nPSNR {psnr(truth, obs):.1f} dB vs. ground truth"),
            (axes[0, 2], est, f"(c) Output: continued {HEIGHT:.0f} m downward\nMultiscale U-Net, PSNR {psnr(truth, est):.1f} dB"),
            (axes[1, 0], truth, "(d) True field on the ground")]
    for ax, img, title in maps:
        im = ax.imshow(img, origin="lower", extent=ext, cmap="RdBu_r", norm=norm)
        ax.axhline(y_prof, color="k", lw=0.8, ls="--")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.85, label="gravity anomaly (mGal)")

    ax = axes[1, 1]
    err = est - truth
    elim = np.abs(err).max()
    im = ax.imshow(err, origin="lower", extent=ext, cmap="RdBu_r", vmin=-elim, vmax=elim)
    ax.set_title(f"(e) Error (c) - (d)\nRMS {np.sqrt(np.mean(err ** 2)):.3f} mGal, range of (d) {np.ptp(truth):.2f} mGal")
    fig.colorbar(im, ax=ax, shrink=0.85, label="mGal")

    ax = axes[1, 2]
    ax.plot(x_km, truth[PROFILE_ROW], "k", lw=2, label="true (ground)")
    ax.plot(x_km, obs[PROFILE_ROW], color="0.55", ls="--", lw=1.5, label=f"observed ({HEIGHT:.0f} m)")
    ax.plot(x_km, est[PROFILE_ROW], color="#d95f02", lw=1.5, label="downward continued")
    ax.set_xlim(ext[0], ext[1])
    ax.set_xlabel("x (km)")
    ax.set_ylabel("mGal")
    ax.set_title(f"(f) Profile at y = {y_prof:.3f} km (dashed line)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    for a in list(axes.ravel())[:5]:
        a.set_xlabel("x (km)")
        a.set_ylabel("y (km)")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    m, bodies = density_model()
    t = time.time()
    obs = grid_gravity(m, CELL, height=HEIGHT)
    truth = grid_gravity(m, CELL, height=0.0)
    print(f"forward modelling: {time.time() - t:.0f} s")
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    save_grid(os.path.join(HERE, "data", "example_obs_h100m.csv"), obs)
    save_grid(os.path.join(HERE, "data", "example_truth_h0m.csv"), truth)

    t = time.time()
    est = downward_continue(obs, spacing=CELL, height=HEIGHT)  # Multiscale U-Net, field weights
    print(f"downward continuation: {time.time() - t:.1f} s")
    print(f"PSNR vs. the true ground-level field (without the outer 4 nodes): observed {psnr(truth, obs):.2f} dB, "
          f"downward continued {psnr(truth, est):.2f} dB")
    os.makedirs(os.path.join(REPO, "docs"), exist_ok=True)
    figure(m, bodies, obs, est, truth, os.path.join(REPO, "docs", "new_grid_example.png"))
    print("figure: docs/new_grid_example.png")


if __name__ == "__main__":
    main()
