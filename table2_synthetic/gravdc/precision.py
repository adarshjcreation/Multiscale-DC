"""Numerical precision of the learned methods.

On Ampere and newer GPUs PyTorch runs float32 convolutions in TF32 by default (10-bit mantissa). At the
PSNR levels of Table 2 (50-65 dB) that changes the 3rd-4th decimal of the PSNR, and the result then also
depends on the GPU model and on the cuDNN algorithm that is picked. ``gravdc`` therefore switches to full
FP32 with deterministic cuDNN when it is imported; Table 2 was computed with this setting.
``set_precision("tf32")`` restores the PyTorch defaults (PSNR then differs by up to about 0.06 dB).
"""
import torch


def _set_conv_tf32(enabled):
    conv = getattr(torch.backends.cudnn, "conv", None)
    if conv is not None and hasattr(conv, "fp32_precision"):  # PyTorch >= 2.9
        conv.fp32_precision = "tf32" if enabled else "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    else:  # older PyTorch
        torch.backends.cudnn.allow_tf32 = enabled
        torch.backends.cuda.matmul.allow_tf32 = False


def set_precision(mode="fp32"):
    """``"fp32"``: full FP32, deterministic cuDNN (default, used for Table 2). ``"tf32"``: PyTorch defaults."""
    if mode not in ("fp32", "tf32"):
        raise ValueError(f"mode must be 'fp32' or 'tf32', not {mode!r}")
    full = mode == "fp32"
    _set_conv_tf32(not full)
    torch.backends.cudnn.deterministic = full
    torch.backends.cudnn.benchmark = False
