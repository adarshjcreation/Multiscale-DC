"""Classical downward-continuation baselines: Tikhonov-regularised DC (TRDC) and TTSIDC.

All functions work on single 2-D numpy grids. Spacing ``dx``/``Cell`` is in metres,
continuation depth ``z`` in metres.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift
from scipy.signal import savgol_filter
from scipy.special import factorial

from .metrics import center_crop


def wavenumber(M, N, dx, dy):
    kx = 2 * np.pi * np.arange(-N / 2, N / 2) / (M * dx)
    ky = 2 * np.pi * np.arange(-M / 2, M / 2) / (N * dy)
    kx, ky = np.meshgrid(kx, ky)
    return np.sqrt(kx ** 2 + ky ** 2)


def rms(a, b):
    return np.sqrt(np.sum((a - b) ** 2) / a.size)


# ----------------------------------------------------------------------------- TRDC
def tr_down_continue(data, alpha, perex=(4, 4), z=800, dx=400, dy=400):
    """Tikhonov-regularised downward continuation: filter exp(kz) / (1 + alpha k^2 exp(kz))."""
    m, n = data.shape
    padded = np.pad(data, perex, mode="constant", constant_values=0)
    k = wavenumber(*padded.shape, dx, dy)
    op = np.exp(k * z) / (1 + alpha * k ** 2 * np.exp(k * z))
    down = np.real(ifft2(ifftshift(op * fftshift(fft2(padded)))))
    return down[perex[0]:m + perex[0], perex[1]:n + perex[1]]


def geometric_grid(start, ratio, n):
    """start * ratio**i built by repeated multiplication (bit-identical to the original loops)."""
    out = np.zeros(n)
    a = start
    for i in range(n):
        out[i] = a
        a *= ratio
    return out


def oracle_alpha(data, truth, n=2000, alpha_start=0.1, ratio=1.01, **kw):
    """Oracle TRDC: the alpha minimising the RMS error to the ground truth (central 24 x 24 window)."""
    alphas = geometric_grid(alpha_start, ratio, n)
    errors = [rms(center_crop(tr_down_continue(data, a, **kw)), center_crop(truth)) for a in alphas]
    return alphas[int(np.argmin(errors))]


def lcurve_alpha(data, n=200, alpha_start=1e-5, ratio=1.1, **kw):
    """Standard TRDC: alpha at the maximum curvature of the log-log L-curve."""
    alphas = geometric_grid(alpha_start, ratio, n)
    res, sol = np.zeros(n), np.zeros(n)
    for i, a in enumerate(alphas):
        down = tr_down_continue(data, a, **kw)
        res[i] = np.linalg.norm(data - down)
        sol[i] = np.linalg.norm(down)
    log_r, log_s = np.log10(res), np.log10(sol)
    dx, dy = np.gradient(log_r), np.gradient(log_s)
    d2x, d2y = np.gradient(dx), np.gradient(dy)
    curvature = np.abs(dx * d2y - dy * d2x) / (dx ** 2 + dy ** 2) ** 1.5
    return alphas[int(np.argmax(curvature))]


# ----------------------------------------------------------------------------- TTSIDC
def laplacian_of_gaussian(f, flag, cell, sigma, k):
    """Second derivative (d2/dx2 if flag == 1 else d2/dy2) via a zero-mean Gaussian-derivative kernel."""
    x, y = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1), indexing="ij")
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / sigma ** 4
    kernel = (x ** 2 - sigma ** 2) * g if flag == 1 else (y ** 2 - sigma ** 2) * g
    kernel = (kernel - kernel.sum() / (2 * k + 1) ** 2) / cell ** 2
    U = np.pad(f, pad_width=k, mode="symmetric")
    out = np.einsum("ijab,ab->ij", sliding_window_view(U, kernel.shape), kernel)
    return out.astype(f.dtype)


def _extend(f, expand):
    m, n = f.shape
    size = int(np.floor(max(m, n) + max(m, n) * expand / 100))
    size += size % 2
    t1, t3 = (size - m) // 2, (size - n) // 2
    grid = np.zeros((size, size))
    grid[t1:t1 + m, t3:t3 + n] = f
    return grid, (t1, t1 + m, t3, t3 + n)


def frequency_integration(f, expand, cell):
    """Vertical integration (division by |k|) in the Fourier domain."""
    fexp, t = _extend(f, expand)
    K = wavenumber(*fexp.shape, cell, cell)
    zero = K == 0
    K[zero] = np.mean(K[~zero])
    op = np.zeros_like(K)
    op[~zero] = 1 / K[~zero]
    out = np.real(ifft2(fftshift(fftshift(fft2(fexp)) * op)))
    return out[t[0]:t[1], t[2]:t[3]]


def up_continue(f, expand, cell, z):
    fexp, t = _extend(f, expand)
    K = wavenumber(*fexp.shape, cell, cell)
    out = np.real(ifft2(ifftshift(fftshift(fft2(fexp)) * np.exp(-z * K))))
    return out[t[0]:t[1], t[2]:t[3]]


def taylor_downward(f, cell, h, expand, order, sigma, window):
    """Truncated Taylor-series downward continuation (vertical derivatives from Laplacians)."""
    k = window // 2
    lap = lambda g: -(laplacian_of_gaussian(g, 1, cell, sigma, k) + laplacian_of_gaussian(g, 2, cell, sigma, k))
    out = np.zeros(f.shape)
    for ind in range(order):
        if ind == 0:
            derivative = f
        elif ind == 1:
            derivative = odd = lap(frequency_integration(f, expand, cell))
        elif ind == 2:
            derivative = even = lap(f)
        elif ind % 2 == 0:
            derivative = even = lap(even)
        else:
            derivative = odd = lap(odd)
        out += (h ** ind / factorial(ind)) * derivative
    return out


def ttsidc(f, cell=400, z=800, expand=10, sigma=1.7, window=3, taylor_down=6, taylor_up=3, iterations=10,
           return_fit=False):
    """Truncated Taylor Series Iterative Downward Continuation (Zhang et al., 2013)."""
    if window % 2 == 0:
        raise ValueError("window must be odd")
    est = taylor_downward(f, cell, z, expand, taylor_down, sigma, window)
    fit = np.zeros(iterations)
    for i in range(iterations):
        est += taylor_downward(f - up_continue(est, expand, cell, z), cell, z, expand, taylor_up, sigma, window)
        if return_fit:
            fit[i] = np.sqrt(np.sum((up_continue(est, expand, cell, z) - f) ** 2) / np.sum(f ** 2))
    return (est, fit) if return_fit else est


def select_sigma_cnorm(f, n=50, sigma_start=1.2, ratio=1.02, iterations=4, **kw):
    """C-norm criterion: sigma at the maximum curvature of log ||u(sigma*ratio) - u(sigma)||_inf vs log sigma."""
    sigmas = geometric_grid(sigma_start, ratio, n)
    cnorm = np.array([
        np.max(np.abs(ttsidc(f, sigma=s * ratio, iterations=iterations, **kw) - ttsidc(f, sigma=s, iterations=iterations, **kw)))
        for s in sigmas
    ])
    logx, logc = np.log(sigmas), np.log(cnorm)
    d1 = np.gradient(logc, logx)
    d2 = np.gradient(d1, logx)
    curvature = np.abs(d2) / (1 + d1 ** 2) ** 1.5
    return sigmas[int(np.argmax(curvature))]


def select_iterations(f, sigma, max_iter=100, smooth_win=11, poly=3, trim=5, kmin=10, **kw):
    """Knee of the (smoothed) log data-misfit curve versus iteration number."""
    _, fit = ttsidc(f, sigma=sigma, iterations=max_iter, return_fit=True, **kw)
    y = savgol_filter(np.log(np.maximum(fit, np.finfo(float).eps)), smooth_win, poly)
    x = np.arange(len(fit))
    dx, dy = np.gradient(x), np.gradient(y)
    d2x, d2y = np.gradient(dx), np.gradient(dy)
    curvature = np.abs(dx * d2y - dy * d2x) / (dx ** 2 + dy ** 2) ** 1.5
    start = max(trim, kmin)
    return int(np.argmax(curvature[start:len(fit) - trim]) + start)
