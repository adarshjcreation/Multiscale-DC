"""Downward continuation of new gravity grids with the pretrained multiscale networks.

    from multiscale_dc import downward_continue
    g0 = downward_continue(g_obs, spacing=50.0, height=100.0)   # g_obs: 2-D array, e.g. in mGal

Command line: ``python -m multiscale_dc grid.csv --spacing 50 --height 100 --plot result.png``.
See the section "Use the networks on your own data" of the README for the inputs, outputs and options.

How it works (the networks were trained on 32 x 32 grids):
1. The grid is extended by 16 nodes on every side (ramping to its median level, against edge effects) and cut
   into overlapping 32 x 32 tiles.
2. Each tile is scaled the way the training grids were: the ground-level field to [-1, 1]. That range is not
   known in advance, so it is estimated from the observations and refined over a few passes.
3. The multiscale routine of the paper continues each tile downward in steps of 0.05 grid spacings.
4. The tiles are scaled back to the unit of the input, blended with smooth (Hann) weights and cropped to the
   original grid.
"""
import os
import sys
import warnings

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "table3_field"))  # networks and operators of the paper (same in both folders)
from gravdc.physics import apply_kernel, continuation_kernel, padded_size  # noqa: E402
from gravdc.sampling import multiscale_dc  # noqa: E402
from gravdc.weights import load_model  # noqa: E402

TILE = 32        # the networks work on 32 x 32 grids
STEP = 0.05      # one network step = upward continuation by 0.05 grid spacings
MAX_STEPS = 40   # trained for 1 to 40 steps, i.e. continuation distances up to 2 grid spacings
WEIGHTS = {      # weight set: folder whose weights/ holds the .pth files
    "field": "table3_field",          # trained on density contrasts -1.0, -0.5, +0.5, +1.0 g/cm^3
    "synthetic": "table2_synthetic",  # trained on +1.0 g/cm^3 only (positive anomalies)
}
DEFAULT_PASSES = {"multiscale_unet": 2, "multiscale_dit": 5}  # scaling passes; chosen in tests (README)
ROUTINES = ("stabilized", "recursive", "single-shot")

__all__ = ["downward_continue", "load_grid", "save_grid", "plot_result", "upward_continue_grid"]

_NETS = {}


class _Operator:
    """A(x, t): t upward continuations of 32 x 32 tiles by STEP grid spacings each, with the zero padding to
    36 x 36 used in training. Only the ratio step / spacing matters, so one operator serves every spacing."""

    def __init__(self, steps, device):
        self.steps = steps
        size = padded_size(TILE, TILE)
        kernel = continuation_kernel(STEP, size, dx=1.0, dy=1.0)
        self.kernel = torch.as_tensor(kernel, dtype=torch.float32, device=device)

    @torch.no_grad()
    def __call__(self, x, t=None):
        for _ in range(self.steps if t is None else t):
            x = apply_kernel(x, self.kernel)
        return x


def _network(model, weights, device):
    key = (model, weights, str(device))
    if key not in _NETS:
        wdir = os.path.join(_REPO, WEIGHTS[weights], "weights")
        try:
            _NETS[key] = load_model(model, device, weights_dir=wdir)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{e}\nDownload the weights first: python scripts/download_weights.py") from None
    return _NETS[key]


def _starts(n, stride):
    s = list(range(0, n - TILE + 1, stride))
    if s[-1] != n - TILE:
        s.append(n - TILE)
    return s


def _continue_tiles(net, A, tiles, passes, routine, device):
    """Downward-continue a stack of tiles (N, 32, 32), in the unit of the input.

    The networks were trained on pairs x = a (g - lo) - 1 in [-1, 1] (g: ground-level field, lo/hi its range,
    a = 2 / (hi - lo)) and y = A(x) = a A(g - c) + (a (c - lo) - 1) A(1), where A pads with zeros. We use
    A(g - c) ~ observation - c with c the median of the tile border, i.e. the field is assumed to continue at its
    border level outside the tile. Adding a constant to the data therefore shifts the result by that constant.
    """
    lo = tiles.min(axis=(1, 2))  # first guess of the ground-level range: the range of the observations
    hi = tiles.max(axis=(1, 2))
    edge = np.concatenate([tiles[:, 0, :], tiles[:, -1, :], tiles[:, 1:-1, 0], tiles[:, 1:-1, -1]], axis=1)
    c = np.median(edge, axis=1)
    floor = 1e-9 * max(np.ptp(tiles), np.finfo(float).tiny)  # flat tiles
    ones = A(torch.ones((len(tiles), 1, TILE, TILE), device=device))  # a constant level, seen by the operator
    for _ in range(passes):
        a = 2.0 / np.maximum(hi - lo, floor)
        y = torch.as_tensor(a[:, None, None] * (tiles - c[:, None, None]), dtype=torch.float32, device=device)
        k = torch.as_tensor(a * (c - lo) - 1.0, dtype=torch.float32, device=device)[:, None, None, None]
        xn = multiscale_dc(net, y[:, None] + k * ones, A, routine)
        est = (xn[:, 0].double().cpu().numpy() + 1.0) / a[:, None, None] + lo[:, None, None]
        smooth = gaussian_filter(est, sigma=(0, 1, 1))  # range of the estimate, without single-node spikes
        lo, hi = smooth.min(axis=(1, 2)), smooth.max(axis=(1, 2))
    return est


def downward_continue(grid, spacing, height, model="multiscale_unet", weights="field", routine="stabilized",
                      passes=None, stride=8, pad=16, device=None, batch_size=256):
    """Downward-continue a gridded gravity anomaly by ``height``.

    grid       2-D array (ny, nx): the anomaly on a regular grid at the observation level, at least 32 x 32
               nodes, same spacing in x and y, no gaps (NaN). Any unit (mGal, um/s^2); the result has the same.
    spacing    grid spacing (any length unit, e.g. m).
    height     continuation distance, in the unit of ``spacing``: the observations lie ``height`` above the level
               of the result. At most 2 x spacing (what the networks were trained for), in steps of
               0.05 x spacing (other values are rounded).
    model      "multiscale_unet" (default) or "multiscale_dit".
    weights    "field" (default; trained on positive and negative density contrasts) or "synthetic" (positive
               contrasts only, 50 m grids of Table 2).
    routine    "stabilized" (default), "recursive" or "single-shot" (one network evaluation; fastest).
    passes     scaling passes (default: 2 for the U-Net, 5 for the DiT); see the module docstring.
    stride     step between the 32 x 32 tiles in nodes (1-32, default 8). Smaller = smoother blending, slower.
    pad        the grid is extended by ``pad`` nodes on every side before the continuation (values ramping
               linearly from the edge to the median of the grid) and cropped back afterwards; this reduces edge
               effects. Default 16; 0 switches it off.
    device     "cuda" or "cpu" (default: cuda if available).
    batch_size tiles per network call.

    Returns the downward-continued grid, a numpy array with the shape and the unit of ``grid``.
    """
    g = np.asarray(grid, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError(f"grid must be a 2-D array, not {g.ndim}-D")
    if min(g.shape) < TILE:
        raise ValueError(f"grid must be at least {TILE} x {TILE} nodes, not {g.shape[0]} x {g.shape[1]}")
    if not np.all(np.isfinite(g)):
        raise ValueError("grid contains NaN or inf values: fill the gaps first (e.g. by interpolation)")
    if spacing <= 0 or height <= 0:
        raise ValueError("spacing and height must be positive")
    if model not in DEFAULT_PASSES:
        raise ValueError(f"model must be one of {sorted(DEFAULT_PASSES)}, not {model!r}")
    if weights not in WEIGHTS:
        raise ValueError(f"weights must be one of {sorted(WEIGHTS)}, not {weights!r}")
    if routine not in ROUTINES:
        raise ValueError(f"routine must be one of {ROUTINES}, not {routine!r}")
    if not 1 <= stride <= TILE:
        raise ValueError(f"stride must be between 1 and {TILE}")
    steps = int(round(height / spacing / STEP))
    if steps > MAX_STEPS:
        raise ValueError(f"height = {height / spacing:.3g} grid spacings, but the networks were trained for at most "
                         f"{MAX_STEPS * STEP:g}. Resample the grid to a spacing of at least {height / 2:g} first "
                         "(see the README).")
    if steps < 1:
        raise ValueError(f"height must be at least {STEP / 2:g} grid spacings")
    if abs(steps * STEP * spacing - height) > 1e-6 * height:
        warnings.warn(f"height rounded to {steps * STEP * spacing:g} ({steps} steps of {STEP} grid spacings)")
    passes = DEFAULT_PASSES[model] if passes is None else int(passes)
    if passes < 1:
        raise ValueError("passes must be at least 1")
    if pad < 0:
        raise ValueError("pad must be 0 or positive")
    if np.ptp(g) == 0:
        return g.copy()  # a constant field stays constant
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    net = _network(model, weights, device)
    A = _Operator(steps, device)
    shape = g.shape
    if pad:
        g = np.pad(g, int(pad), mode="linear_ramp", end_values=float(np.median(g)))
    corners = [(r, c) for r in _starts(g.shape[0], stride) for c in _starts(g.shape[1], stride)]
    tiles = np.stack([g[r:r + TILE, c:c + TILE] for r, c in corners])
    est = np.concatenate([_continue_tiles(net, A, tiles[i:i + batch_size], passes, routine, device)
                          for i in range(0, len(tiles), batch_size)])
    w1 = np.hanning(TILE + 2)[1:-1]
    w = np.outer(w1, w1)  # tile weights: largest in the centre, where the networks are most accurate
    out, wsum = np.zeros_like(g), np.zeros_like(g)
    for (r, c), e in zip(corners, est):
        out[r:r + TILE, c:c + TILE] += w * e
        wsum[r:r + TILE, c:c + TILE] += w
    out /= wsum
    if pad:
        out = out[pad:pad + shape[0], pad:pad + shape[1]]
    return out


def upward_continue_grid(grid, spacing, height):
    """Upward continuation of a grid by ``height`` (FFT, exp(-|k| h), with 10 % zero padding). Useful to check a
    result: upward-continuing the downward-continued grid should give back the observations."""
    g = torch.as_tensor(np.asarray(grid, dtype=np.float64))[None, None]
    size = padded_size(*g.shape[-2:])
    kernel = torch.as_tensor(continuation_kernel(height, size, dx=spacing, dy=spacing))
    lo = g.mean()
    return (apply_kernel(g - lo, kernel) + lo)[0, 0].numpy()


def load_grid(path, var=None):
    """Read a grid: .npy, .csv / .txt / .dat (a 2-D table of values; comma, semicolon or whitespace separated)
    or .mat (variable ``var``; a vector of n^2 values is reshaped row by row to n x n, as in our datasets)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        g = np.load(path)
    elif ext in (".csv", ".txt", ".dat", ".asc"):
        with open(path) as f:
            sample = f.read(4096)
        delim = "," if "," in sample else (";" if ";" in sample else None)
        g = np.loadtxt(path, delimiter=delim, ndmin=2)
    elif ext == ".mat":
        import scipy.io

        data = {k: v for k, v in scipy.io.loadmat(path).items() if not k.startswith("__")}
        if var is None:
            cands = [k for k, v in data.items() if isinstance(v, np.ndarray) and v.dtype.kind in "fiu" and v.size >= TILE ** 2]
            if len(cands) != 1:
                raise ValueError(f"{path}: choose the variable with --var (numeric variables: {sorted(cands)})")
            var = cands[0]
        g = np.asarray(data[var], dtype=float)
        if g.ndim == 2 and 1 in g.shape:
            n = int(round(np.sqrt(g.size)))
            if n * n != g.size:
                raise ValueError(f"{path}: variable {var!r} is a vector of {g.size} values, not a square grid")
            g = g.reshape(n, n)
    else:
        raise ValueError(f"unknown file type {ext!r}: use .npy, .csv/.txt or .mat")
    g = np.squeeze(np.asarray(g, dtype=float))
    if g.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D grid, got shape {g.shape}")
    return g


def save_grid(path, grid):
    """Write a grid as .npy or .csv / .txt (comma separated)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        np.save(path, grid)
    elif ext in (".csv", ".txt", ".dat"):
        np.savetxt(path, grid, delimiter=",", fmt="%.8g")
    else:
        raise ValueError(f"unknown output type {ext!r}: use .npy or .csv")


def plot_result(observed, continued, spacing, height, path=None, unit="mGal", truth=None):
    """Maps of the observations and the downward-continued grid (and the true field and the error, if given)."""
    import matplotlib

    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(observed, f"Observed (height {height:g})"), (continued, "Downward continued")]
    if truth is not None:
        panels += [(truth, "True field"), (continued - truth, "Error (continued - true)")]
    ny, nx = observed.shape
    ext = [0, nx * spacing, 0, ny * spacing]
    vmin, vmax = np.min(continued), np.max(continued)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8), constrained_layout=True)
    for ax, (img, title) in zip(np.atleast_1d(axes), panels):
        if title.startswith("Error"):
            lim = np.max(np.abs(img))
            im = ax.imshow(img, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim)
        else:
            im = ax.imshow(img, origin="lower", extent=ext, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.8, label=unit)
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig
