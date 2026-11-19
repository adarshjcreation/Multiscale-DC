"""Upward-continuation forward operators used in the paper.

Two forward models appear in Table 2:

* ``upward_continue(x, 100)`` - a single 100 m continuation. Used to create the
  observations for DU-Net, DC-Net, U-Net, Physics-trained, CNN-regularized (PnP)
  and denoising diffusion.
* ``MultiscaleOperator`` - 40 successive 2.5 m continuations (total 100 m), each
  with zero padding to 36 x 36 and cropping back to 32 x 32. Used by the
  multiscale models (DiT / U-Net) and by the classical baselines (TTSIDC, TRDC).
"""
import numpy as np
import torch
from torch.fft import fft2, ifft2, fftshift, ifftshift

STATION_SPACING = 50.0  # m
LINE_SPACING = 50.0     # m
EXPAND = 10             # zero-padding in percent (32 -> 36 nodes)


def wavenumber(m, n, dx=STATION_SPACING, dy=LINE_SPACING):
    """Radial wavenumber |k| on an m x n grid (fft-shifted ordering)."""
    kx = 2 * np.pi / (dx * n) * np.arange(-n / 2, n / 2)
    ky = 2 * np.pi / (dy * m) * np.arange(-m / 2, m / 2)
    KX, KY = np.meshgrid(kx, ky)
    return np.sqrt(KX ** 2 + KY ** 2)


def padded_size(m, n, expand=EXPAND):
    size = int(np.floor(max(m, n) * (1 + expand / 100.0)))
    return size + (size % 2)


def continuation_kernel(height, size, dx=STATION_SPACING, dy=LINE_SPACING):
    """Fourier-domain upward-continuation filter exp(-|k| h) on a size x size grid."""
    return np.exp(-wavenumber(size, size, dx, dy) * height)


def apply_kernel(x, kernel):
    """Zero-pad x (B, C, m, n), filter with ``kernel`` (padded size) in Fourier domain, crop back."""
    m, n = x.shape[-2:]
    size = kernel.shape[-1]
    top, left = (size - m) // 2, (size - n) // 2
    xp = torch.nn.functional.pad(x, (left, size - n - left, top, size - m - top))
    U = fftshift(fft2(xp), dim=(-2, -1)) * kernel
    out = torch.real(ifft2(ifftshift(U, dim=(-2, -1))))
    return out[..., top:top + m, left:left + n]


@torch.no_grad()
def upward_continue(x, height, dx=STATION_SPACING, dy=LINE_SPACING):
    """Single-shot upward continuation of a batch x (B, C, 32, 32) by ``height`` metres."""
    size = padded_size(*x.shape[-2:])
    kernel = torch.as_tensor(continuation_kernel(height, size, dx, dy), dtype=torch.float32, device=x.device)
    return apply_kernel(x.float(), kernel)


class MultiscaleOperator:
    """A^t: t successive upward continuations by ``step`` metres (default 40 x 2.5 m = 100 m)."""

    def __init__(self, steps=40, step=2.5, grid=32, device="cpu"):
        self.steps = steps
        size = padded_size(grid, grid)
        self.kernel = torch.as_tensor(continuation_kernel(step, size), dtype=torch.float32, device=device)

    @torch.no_grad()
    def __call__(self, x, t=None):
        """Apply the elementary 2.5 m continuation ``t`` times (default: all ``steps``)."""
        for _ in range(self.steps if t is None else t):
            x = apply_kernel(x, self.kernel)
        return x
