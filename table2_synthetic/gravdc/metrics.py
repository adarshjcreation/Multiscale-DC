"""Evaluation metrics (Table 2): PSNR, SSIM, relative error and RMSE on the central 24 x 24 window."""
import json
import os

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

CROP = 24  # the outer 4 nodes on each side are excluded (edge effects of the FFT operators)


def center_crop(img, size=CROP):
    c, h = img.shape[0] // 2, size // 2
    return img[c - h:c + h, c - h:c + h]


def image_metrics(gt, pred, crop=CROP):
    """Metrics of one reconstruction. ``gt`` and ``pred`` are 2-D numpy arrays."""
    gt, pred = np.asarray(gt), np.asarray(pred)
    if crop:
        gt, pred = center_crop(gt, crop), center_crop(pred, crop)
    data_range = gt.max() - gt.min()
    return {
        "PSNR": peak_signal_noise_ratio(gt, pred, data_range=data_range),
        "SSIM": structural_similarity(gt, pred, data_range=data_range),
        "RE": np.linalg.norm(gt - pred) / np.linalg.norm(gt),
        "RMSE": np.sqrt(np.mean((gt - pred) ** 2)),
    }


def evaluate(gt, pred, crop=CROP):
    """Per-image metrics for batches gt, pred of shape (N, 1, H, W) (tensors or arrays)."""
    gt, pred = _numpy(gt), _numpy(pred)
    return [image_metrics(g[0], p[0], crop) for g, p in zip(gt, pred)]


def summarize(per_image):
    """Mean and (population) standard deviation over images, as reported in the paper."""
    keys = per_image[0].keys()
    return {k: (float(np.mean([m[k] for m in per_image])), float(np.std([m[k] for m in per_image]))) for k in keys}


def format_summary(summary):
    return "  ".join(f"{k}: {m:.4f} ± {s:.4f}" for k, (m, s) in summary.items())


def save_result(name, per_image, names, folder, **extra):
    """Store one Table-2 row as ``<folder>/<name>.json`` (read by 00_Reproduce_Table2.ipynb)."""
    os.makedirs(folder, exist_ok=True)
    record = {
        "method": name,
        "summary": summarize(per_image),
        "per_image": {n: {k: float(v) for k, v in m.items()} for n, m in zip(names, per_image)},
        **extra,
    }
    with open(os.path.join(folder, f"{name}.json"), "w") as f:
        json.dump(record, f, indent=2)


def _numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
