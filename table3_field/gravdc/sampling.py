"""Inference procedures for the iterative / generative methods of Table 3."""
import torch

from .physics import upward_continue


@torch.no_grad()
def multiscale_dc(model, y, A, routine="stabilized"):
    """Multiscale downward continuation with a time-conditioned restoration network.

    model   : network R(x_t, t) predicting the ground-level field x_0 from data continued t steps upward
    y       : observations x_T = A^T x_0, shape (B, 1, 32, 32)
    A       : ``MultiscaleOperator``; A(x, t) applies t elementary upward continuations
    routine : 'single-shot' - x_0 = R(y, T) (one network evaluation)
              'stabilized'  - x_{t-1} = x_t - A^t R(x_t, t) + A^{t-1} R(x_t, t)   (Stabilized DC)
              'recursive'   - x_{t-1} = A^{t-1} R(x_t, t)                         (Recursive DC)
    Returns the reconstruction x_0.
    """
    T = A.steps
    x = y
    for t in range(T, 0, -1):
        step = torch.full((x.shape[0],), t - 1, dtype=torch.long, device=x.device)
        x0 = model(x, step)
        if routine == "single-shot":
            return x0
        if routine == "stabilized":
            x = x - A(x0, t) + A(x0, t - 1)
        elif routine == "recursive":
            x = A(x0, t - 1)
        else:
            raise ValueError(f"unknown routine {routine!r}")
    return x


def cosine_beta_schedule(timesteps, s=0.008):
    import numpy as np

    x = np.linspace(0, timesteps, timesteps + 1)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(torch.from_numpy(betas), 0.0001, 0.9999)


@torch.no_grad()
def ddpm_sample(model, y, timesteps=1000, steps=250, n_samples=10, seed=0):
    """Conditional denoising-diffusion sampler of the 'Denoising diffusion (U-Net)' baseline.

    The noise predictor ``model(x, t, cond=y)`` was trained with a 1000-step cosine schedule;
    inference runs the last ``steps`` reverse steps from pure noise. 250 steps were chosen on training
    grids; with 500 steps some samples diverge, and with 1000 steps all of them do.
    All random numbers come from a CPU generator seeded with ``seed`` and are then moved to the
    device, so the noise is identical on every GPU and on CPU.
    Returns a tensor of shape (n_samples, B, 1, 32, 32).
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def randn(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(y.device)

    model.eval()
    betas = cosine_beta_schedule(timesteps)
    alphas_cumprod = torch.cumprod(1.0 - betas, 0).float().to(y.device)  # schedule held in float32, as in training
    betas = betas.float().to(y.device)
    samples = []
    for _ in range(n_samples):
        x = randn(y.shape)
        for t_inv in reversed(range(steps)):
            t = torch.full((y.shape[0],), t_inv, device=y.device, dtype=torch.long)
            noise_pred = torch.nan_to_num(model(x, t, y), nan=0.0, posinf=1e2, neginf=-1e2)
            alpha = alphas_cumprod[t][:, None, None, None]
            beta = betas[t][:, None, None, None]
            sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=1e-8))
            sqrt_one_minus_alpha = torch.sqrt(torch.clamp(1.0 - alpha, min=1e-8))
            x = (x - (1.0 - alpha) / sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
            if t_inv > 0:
                x = x + randn(x.shape) * torch.sqrt(torch.clamp(beta, min=1e-8))
                x = torch.clamp(x, -10.0, 10.0)
        samples.append(torch.clamp(x, -1.0, 1.0).float())
    return torch.stack(samples)


@torch.no_grad()
def pnp_admm(y, denoiser, height=800.0, iterations=100, lam=2.0):
    """CNN-regularized downward continuation: plug-and-play ADMM with a learned denoiser.

    x-update is one gradient-like step on the data term (x = z - u - lam (A(z - u) - y)),
    z-update is the denoiser, u is the scaled dual variable.
    """
    A = lambda v: upward_continue(v, height)
    x = torch.zeros_like(y)
    z = x.clone()
    u = torch.zeros_like(x)
    for _ in range(iterations):
        x_tilde = z - u
        x = x_tilde - lam * (A(x_tilde) - y)
        z = denoiser(x + u, None)
        u = u + x - z
    return z
