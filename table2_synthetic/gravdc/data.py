"""Loading and normalising the synthetic gravity test grids."""
import os

import numpy as np
import scipy.io
import torch

GRID = 32  # grids are 32 x 32 nodes


def scale_to_minus1_1(x):
    """Per-grid min-max scaling to [-1, 1] (same as the training pipeline)."""
    lo, hi = x.min(), x.max()
    if hi > lo:
        return 2 * (x - lo) / (hi - lo) - 1
    return x


def load_grid(path):
    """Load one ``.mat`` file (variable ``d``) as a float32 tensor of shape (1, 32, 32) in [-1, 1]."""
    d = scipy.io.loadmat(path)["d"].reshape(GRID, GRID).astype(np.float32)
    return scale_to_minus1_1(torch.from_numpy(d).unsqueeze(0))


def load_test_set(folder):
    """Return (file names, tensor of shape (N, 1, 32, 32)) for every ``.mat`` file in ``folder``, sorted by name."""
    names = sorted(f for f in os.listdir(folder) if f.endswith(".mat"))
    grids = torch.stack([load_grid(os.path.join(folder, f)) for f in names])
    return names, grids
