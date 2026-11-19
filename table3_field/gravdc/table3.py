"""Table 3 of the paper: mean ± standard deviation over the 10 field test patches
(central 24 x 24 window, networks in full FP32). Each notebook prints its result next to these values."""

ORDER = [
    "TTSIDC", "Oracle TRDC", "TRDC", "DU-Net", "DC-Net", "Physics-Trained", "CNN-regularized", "U-Net",
    "Denoising diffusion (U-Net)", "Multiscale DiT (Single-shot)", "Multiscale DiT (Stabilized DC)",
    "Multiscale U-Net (Single-shot)", "Multiscale U-Net (Stabilized DC)", "Multiscale U-Net (Recursive DC)",
]
METRICS = ["PSNR", "SSIM", "RE", "RMSE"]

TABLE3 = {
    "TTSIDC":                           {"PSNR": (36.8247, 2.5611), "SSIM": (0.9882, 0.0076), "RE": (0.0746, 0.0243), "RMSE": (0.0262, 0.0097)},
    "Oracle TRDC":                      {"PSNR": (40.9487, 2.7437), "SSIM": (0.9910, 0.0051), "RE": (0.0455, 0.0125), "RMSE": (0.0160, 0.0055)},
    "TRDC":                             {"PSNR": (37.7190, 2.1283), "SSIM": (0.9862, 0.0084), "RE": (0.0656, 0.0176), "RMSE": (0.0231, 0.0080)},
    "DU-Net":                           {"PSNR": (36.2222, 1.8690), "SSIM": (0.9861, 0.0088), "RE": (0.0768, 0.0144), "RMSE": (0.0270, 0.0073)},
    "DC-Net":                           {"PSNR": (33.9575, 1.9905), "SSIM": (0.9771, 0.0116), "RE": (0.1003, 0.0208), "RMSE": (0.0353, 0.0101)},
    "Physics-Trained":                  {"PSNR": (26.4089, 1.7020), "SSIM": (0.8490, 0.1073), "RE": (0.2391, 0.0506), "RMSE": (0.0816, 0.0128)},
    "CNN-regularized":                  {"PSNR": (28.3958, 1.7156), "SSIM": (0.9254, 0.0316), "RE": (0.1888, 0.0293), "RMSE": (0.0663, 0.0158)},
    "U-Net":                            {"PSNR": (32.4771, 1.9784), "SSIM": (0.9724, 0.0142), "RE": (0.1201, 0.0277), "RMSE": (0.0421, 0.0119)},
    "Denoising diffusion (U-Net)":      {"PSNR": (38.4493, 2.7439), "SSIM": (0.9911, 0.0089), "RE": (0.0618, 0.0214), "RMSE": (0.0218, 0.0091)},
    "Multiscale DiT (Single-shot)":     {"PSNR": (39.8545, 2.6601), "SSIM": (0.9940, 0.0067), "RE": (0.0525, 0.0181), "RMSE": (0.0186, 0.0078)},
    "Multiscale DiT (Stabilized DC)":   {"PSNR": (39.3045, 2.6594), "SSIM": (0.9933, 0.0073), "RE": (0.0561, 0.0197), "RMSE": (0.0198, 0.0084)},
    "Multiscale U-Net (Single-shot)":   {"PSNR": (41.8093, 3.0735), "SSIM": (0.9953, 0.0069), "RE": (0.0427, 0.0186), "RMSE": (0.0152, 0.0078)},
    "Multiscale U-Net (Stabilized DC)": {"PSNR": (40.8483, 3.0420), "SSIM": (0.9943, 0.0078), "RE": (0.0478, 0.0203), "RMSE": (0.0169, 0.0085)},
    "Multiscale U-Net (Recursive DC)":  {"PSNR": (28.6774, 2.1266), "SSIM": (0.9499, 0.0158), "RE": (0.1836, 0.0362), "RMSE": (0.0641, 0.0156)},
}


def matches(summary, name):
    """True if ``summary`` equals the Table-3 row ``name`` to 4 decimals (mean and std of every metric)."""
    ref = TABLE3[name]
    return all(round(summary[k][0], 4) == ref[k][0] and round(summary[k][1], 4) == ref[k][1] for k in METRICS)


def compare(name, summary):
    """Print this run next to the Table-3 row ``name``."""
    print(name)
    if name not in TABLE3:
        for k in METRICS:
            print(f"  {k:6s} {summary[k][0]:11.4f} ± {summary[k][1]:6.4f}")
        return
    print(f"  {'metric':6s} {'this run':>20s} {'Table 3':>20s}")
    for k in METRICS:
        (m, s), (pm, ps) = summary[k], TABLE3[name][k]
        ok = "✓" if (round(m, 4), round(s, 4)) == (pm, ps) else "✗"
        print(f"  {k:6s} {m:11.4f} ± {s:6.4f} {pm:11.4f} ± {ps:6.4f}  {ok}")
