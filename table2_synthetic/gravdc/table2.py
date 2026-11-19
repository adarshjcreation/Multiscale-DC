"""Table 2 of the paper: mean ± standard deviation over the 10 synthetic test grids
(central 24 x 24 window, networks in full FP32). Each notebook prints its result next to these values."""

ORDER = [
    "TTSIDC", "Oracle TRDC", "TRDC", "DU-Net", "DC-Net", "U-Net", "Physics-Trained", "CNN-regularized",
    "Denoising diffusion (U-Net)", "Multiscale DiT (Single-shot)", "Multiscale DiT (Stabilized DC)",
    "Multiscale U-Net (Single-shot)", "Multiscale U-Net (Stabilized DC)", "Multiscale U-Net (Recursive DC)",
]
METRICS = ["PSNR", "SSIM", "RE", "RMSE"]

TABLE2 = {
    "TTSIDC": {"PSNR": (47.9276, 4.5850), "SSIM": (0.9985, 0.0014), "RE": (0.0135, 0.0077), "RMSE": (0.0094, 0.0067)},
    "Oracle TRDC": {"PSNR": (37.0393, 1.9052), "SSIM": (0.9761, 0.0082), "RE": (0.0424, 0.0040), "RMSE": (0.0282, 0.0066)},
    "TRDC": {"PSNR": (36.3714, 1.7217), "SSIM": (0.9756, 0.0057), "RE": (0.0457, 0.0039), "RMSE": (0.0303, 0.0064)},
    "DU-Net": {"PSNR": (48.6022, 2.0723), "SSIM": (0.9984, 0.0007), "RE": (0.0113, 0.0018), "RMSE": (0.0075, 0.0019)},
    "DC-Net": {"PSNR": (52.3368, 3.9071), "SSIM": (0.9995, 0.0005), "RE": (0.0078, 0.0034), "RMSE": (0.0053, 0.0029)},
    "U-Net": {"PSNR": (57.8891, 3.3056), "SSIM": (0.9999, 0.0001), "RE": (0.0040, 0.0015), "RMSE": (0.0027, 0.0013)},
    "Physics-Trained": {"PSNR": (26.9830, 1.0557), "SSIM": (0.8539, 0.0810), "RE": (0.1378, 0.0346), "RMSE": (0.0878, 0.0101)},
    "CNN-regularized": {"PSNR": (50.7438, 8.5486), "SSIM": (0.9982, 0.0040), "RE": (0.0140, 0.0178), "RMSE": (0.0102, 0.0140)},
    "Denoising diffusion (U-Net)": {"PSNR": (59.0080, 1.6162), "SSIM": (0.9999, 0.0000), "RE": (0.0034, 0.0004), "RMSE": (0.0022, 0.0005)},
    "Multiscale DiT (Single-shot)": {"PSNR": (60.1563, 4.4481), "SSIM": (0.9998, 0.0003), "RE": (0.0033, 0.0020), "RMSE": (0.0023, 0.0017)},
    "Multiscale DiT (Stabilized DC)": {"PSNR": (64.7164, 6.6262), "SSIM": (0.9999, 0.0001), "RE": (0.0023, 0.0022), "RMSE": (0.0017, 0.0018)},
    "Multiscale U-Net (Single-shot)": {"PSNR": (58.9173, 3.9686), "SSIM": (0.9998, 0.0001), "RE": (0.0037, 0.0019), "RMSE": (0.0025, 0.0016)},
    "Multiscale U-Net (Stabilized DC)": {"PSNR": (63.4603, 6.9387), "SSIM": (0.9999, 0.0001), "RE": (0.0028, 0.0029), "RMSE": (0.0020, 0.0023)},
    "Multiscale U-Net (Recursive DC)": {"PSNR": (32.8412, 1.8151), "SSIM": (0.9403, 0.0300), "RE": (0.0723, 0.0245), "RMSE": (0.0452, 0.0083)},
}


def matches(summary, name):
    """True if ``summary`` equals the Table-2 row ``name`` to 4 decimals (mean and std of every metric)."""
    ref = TABLE2[name]
    return all(round(summary[k][0], 4) == ref[k][0] and round(summary[k][1], 4) == ref[k][1] for k in METRICS)


def compare(name, summary):
    """Print this run next to the Table-2 row ``name``."""
    print(name)
    if name not in TABLE2:
        for k in METRICS:
            print(f"  {k:6s} {summary[k][0]:11.4f} ± {summary[k][1]:6.4f}")
        return
    print(f"  {'metric':6s} {'this run':>20s} {'Table 2':>20s}")
    for k in METRICS:
        (m, s), (pm, ps) = summary[k], TABLE2[name][k]
        ok = "✓" if (round(m, 4), round(s, 4)) == (pm, ps) else "✗"
        print(f"  {k:6s} {m:11.4f} ± {s:6.4f} {pm:11.4f} ± {ps:6.4f}  {ok}")
