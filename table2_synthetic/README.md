# Multiscale Learning for Downward Continuation of Gravity Data: Table 2 (synthetic data)

This folder reproduces **Table 2** of the paper, the comparison of downward-continuation approaches on
synthetic data. It contains the code, the 10 synthetic test grids, the pretrained weights (downloaded
separately) and one notebook per method. Running the notebooks regenerates every entry of the table and
checks it against the reference values.

## Contents

```
data/test/                10 synthetic test grids (.mat, variable `d`, 32 x 32 nodes, 50 m spacing)
gravdc/                   Python package shared by the notebooks
  data.py                   loading + per-grid scaling to [-1, 1]
  physics.py                upward-continuation operators (single 100 m step; multiscale 40 x 2.5 m)
  metrics.py                PSNR, SSIM, RE, RMSE on the central 24 x 24 window
  classical.py              TTSIDC and Tikhonov (TRDC) with automatic parameter selection
  sampling.py               multiscale DC sampling, DDPM sampler, plug-and-play ADMM
  models/                   U-Net (ConvNeXt), DiT-L/2, DC-Net, DU-Net
  weights.py                model constructors and weight files
  precision.py              full-FP32, deterministic evaluation (applied on import)
  table2.py                 reference values of Table 2
  plotting.py               quick-look figures
notebooks/
  01_TTSIDC                 02_TRDC (TRDC and Oracle TRDC)      03_DU-Net
  04_DC-Net                 05_U-Net                            06_Physics-Trained
  07_CNN-Regularized        08_Denoising_Diffusion              09_Multiscale_DiT
  10_Multiscale_UNet        00_Reproduce_Table2 (collects and checks all rows)
weights/                  pretrained weights (see weights/README.md)
results/                  one JSON file per row (per-grid metrics) and the assembled table2.md / table2.csv
scripts/run_all.sh        runs all notebooks in order
scripts/check_weights.py  checks the weight files (MD5)
```

## Setup

```bash
pip install -r ../requirements.txt                 # exact package versions (repository root)
python ../scripts/download_weights.py --table 2   # weights from Zenodo into weights/, MD5-checked
python scripts/check_weights.py                    # optional: check the weight files again
```

A CUDA GPU is recommended. Everything also runs on CPU (see Reproducibility).

## Running

```bash
bash scripts/run_all.sh        # or run notebooks 01 ... 10 and then 00 by hand
```

On an RTX 3090 every notebook takes less than 15 s, except `08_Denoising_Diffusion` (about 3.5 minutes:
3 seeds x 10 samples x 250 reverse steps); the whole set takes about 5 minutes. Each notebook prints its
result next to the reference value. Notebook `00` assembles the table, marks every row with ✓ when all four
metrics (mean and standard deviation) equal Table 2 to 4 decimals, and writes `results/table2.md` and
`results/table2.csv`.

## Problem setting and evaluation protocol

* **Ground truth** $x_0$: 10 synthetic test grids of 32 x 32 nodes (50 m spacing), each scaled to [-1, 1].
* **Observation**: $x_0$ continued 100 m upward.
  * Multiscale models, TTSIDC and TRDC: the multiscale operator, 40 continuations of 2.5 m, each with zero
    padding to 36 x 36 and cropping back to 32 x 32.
  * DU-Net, DC-Net, U-Net, Physics-trained, CNN-regularized and denoising diffusion: a single 100 m
    continuation with zero padding to 36 x 36.
* **Metrics**: PSNR (with the data range of the ground truth), SSIM, relative error
  RE = ‖x̂ − x‖ / ‖x‖ and RMSE, computed on the central 24 x 24 window (the outer 4 nodes are affected by
  the FFT edge effects). Every row is the mean ± standard deviation (ddof = 0) over all 10 grids.
* **Precision**: all networks run in full FP32 with deterministic cuDNN.

## Methods

| Approach | Notebook | Weights | Notes |
|---|---|---|---|
| TTSIDC (Zhang et al., 2013) | 01 | – | σ by the C-norm criterion, iterations at the knee of the data-misfit curve (no ground truth) |
| Oracle TRDC (Tikhonov et al., 1968) | 02 | – | α minimising the RMS error to the ground truth (upper bound for TRDC) |
| TRDC (Tikhonov et al., 1968) | 02 | – | α at the corner of the L-curve (no ground truth) |
| DU-Net (Ye et al., 2022) | 03 | `dunet.pth` | |
| DC-Net (Li et al., 2023) | 04 | `dcnet.pth` | |
| Backbone U-Net | 05 | `unet_bansal.pth` | fixed-height U-Net, the backbone of the multiscale models; called `U-Net` in `results/` |
| Physics-Trained (Sun et al., 2025) | 06 | `physics_trained.pth` | fitted directly on the 10 test observations through the physics loss ‖Ax̂ − y‖₁; the ground truth is used only for evaluation |
| CNN-regularized | 07 | `pnp_denoiser.pth` | plug-and-play ADMM, 50 iterations, λ = 1.25 |
| Denoising diffusion (U-Net) | 08 | `ddpm_conditional_unet_raw.pth` | 250 reverse steps (chosen on training grids), metrics on the posterior mean of 10 samples per grid, seed 2 |
| Multiscale DiT (ours) | 09 | `multiscale_dit.pth` | Single-shot and Stabilized DC |
| Multiscale U-Net (ours) | 10 | `multiscale_unet.pth` | Single-shot, Stabilized DC and Recursive DC |

All learned methods are evaluated with their raw training weights (no exponential moving average).

## Table 2

Output of `00_Reproduce_Table2.ipynb` (`results/table2.md`). Bold: best value of each column.

| Approach | PSNR (dB) ↑ | SSIM ↑ | RE ↓ | RMSE ↓ |
|---|---|---|---|---|
| TTSIDC | 47.9276 ± 4.5850 | 0.9985 ± 0.0014 | 0.0135 ± 0.0077 | 0.0094 ± 0.0067 |
| Oracle TRDC | 37.0393 ± 1.9052 | 0.9761 ± 0.0082 | 0.0424 ± 0.0040 | 0.0282 ± 0.0066 |
| TRDC | 36.3714 ± 1.7217 | 0.9756 ± 0.0057 | 0.0457 ± 0.0039 | 0.0303 ± 0.0064 |
| DU-Net | 48.6022 ± 2.0723 | 0.9984 ± 0.0007 | 0.0113 ± 0.0018 | 0.0075 ± 0.0019 |
| DC-Net | 52.3368 ± 3.9071 | 0.9995 ± 0.0005 | 0.0078 ± 0.0034 | 0.0053 ± 0.0029 |
| U-Net | 57.8891 ± 3.3056 | 0.9999 ± 0.0001 | 0.0040 ± 0.0015 | 0.0027 ± 0.0013 |
| Physics-Trained | 26.9830 ± 1.0557 | 0.8539 ± 0.0810 | 0.1378 ± 0.0346 | 0.0878 ± 0.0101 |
| CNN-regularized | 50.7438 ± 8.5486 | 0.9982 ± 0.0040 | 0.0140 ± 0.0178 | 0.0102 ± 0.0140 |
| Denoising diffusion (U-Net) | 59.0080 ± 1.6162 | 0.9999 ± 0.0000 | 0.0034 ± 0.0004 | 0.0022 ± 0.0005 |
| Multiscale DiT (Single-shot) | 60.1563 ± 4.4481 | 0.9998 ± 0.0003 | 0.0033 ± 0.0020 | 0.0023 ± 0.0017 |
| Multiscale DiT (Stabilized DC) | **64.7164 ± 6.6262** | **0.9999 ± 0.0001** | **0.0023 ± 0.0022** | **0.0017 ± 0.0018** |
| Multiscale U-Net (Single-shot) | 58.9173 ± 3.9686 | 0.9998 ± 0.0001 | 0.0037 ± 0.0019 | 0.0025 ± 0.0016 |
| Multiscale U-Net (Stabilized DC) | 63.4603 ± 6.9387 | 0.9999 ± 0.0001 | 0.0028 ± 0.0029 | 0.0020 ± 0.0023 |
| Multiscale U-Net (Recursive DC) | 32.8412 ± 1.8151 | 0.9403 ± 0.0300 | 0.0723 ± 0.0245 | 0.0452 ± 0.0083 |

**Denoising diffusion: seeds and number of reverse steps.** The sampler is stochastic, so its row depends
on the seed. Notebook 08 runs seeds 0, 1 and 2 and reports seed 2:

| Seed | PSNR (dB) | SSIM | RE | RMSE |
|---|---|---|---|---|
| 0 | 58.6030 ± 1.5464 | 0.9999 ± 0.0000 | 0.0035 ± 0.0004 | 0.0023 ± 0.0005 |
| 1 | 58.9006 ± 1.7356 | 0.9999 ± 0.0000 | 0.0034 ± 0.0005 | 0.0023 ± 0.0005 |
| 2 (Table 2) | 59.0080 ± 1.6162 | 0.9999 ± 0.0000 | 0.0034 ± 0.0004 | 0.0022 ± 0.0005 |

The sampler starts from Gaussian noise and runs the last `steps` steps of the 1000-step training schedule.
The number of steps was chosen on 20 training grids (every 1100th file of the training set, seed 0):
the posterior mean reaches 55.98 dB with 250 steps, but only 39.88 dB with 500 steps, where some samples
diverge on half of the grids; with 1000 steps every sample diverges. On the test grids the picture is the
same (posterior mean, seeds 0–2: 58.60–59.01 dB with 250 steps, 52.03–53.58 dB with 500 steps, 8.59 dB
with 1000 steps). The standard DDPM ancestral sampler (1000 steps) and DDIM (100 steps, η = 0) give
35.6–40.0 dB and 28.6–30.4 dB on the test grids; this network has no time embedding, and the sampler
used here, which re-estimates the ground-level field at every step and adds the schedule's noise
√β_t, suits it better.

## Reproducibility

* **Precision.** On Ampere and newer GPUs PyTorch runs float32 convolutions in TF32 by default, which
  changes the 3rd–4th decimal of the PSNR and makes it depend on the GPU and the cuDNN algorithm.
  `gravdc` therefore switches to full FP32 with deterministic cuDNN when it is imported
  (`gravdc/precision.py`).
* **Random numbers.** The only stochastic method, denoising diffusion, draws all its noise from a CPU
  random generator with a fixed seed, so the noise is the same on every device.
* **Checks that were run:**
  * Two complete runs of `scripts/run_all.sh` from an empty `results/` folder (RTX 3090) gave
    bit-identical per-grid metrics for all 14 rows, and all 14 rows match `gravdc/table2.py`.
  * A CPU-only run of notebooks 01–07, 09 and 10: TTSIDC, TRDC, Oracle TRDC, DC-Net, CNN-regularized and
    Physics-Trained agree with Table 2 to 4 decimals; DU-Net and U-Net differ by 0.0001 dB in the PSNR;
    the multiscale rows differ by at most 0.004 dB in the PSNR (and in the last digit of one SSIM). These
    are floating-point rounding differences between CPU and GPU kernels, which the 40-step multiscale
    routines accumulate.
  * Denoising diffusion: a repeated GPU run of all three seeds reproduced the stored per-grid metrics
    exactly. Seed 2 (the Table-2 row) on CPU, 10 grids x 10 samples x 250 steps: the samples agree with the
    GPU samples to 3e-5 and the per-grid PSNR to 0.004 dB; the row is 59.0083 ± 1.6153 dB
    (GPU: 59.0080 ± 1.6162 dB). One seed takes about 22 minutes on a 16-core CPU, so the full notebook 08
    takes about an hour on CPU.
* **Tested environment:** Python 3.11.6, PyTorch 2.9.0 with CUDA 12.8 and cuDNN 9.10, NVIDIA GeForce
  RTX 3090, Linux. The exact package versions are pinned in `../requirements.txt`.

## Citation

See the [repository README](../README.md#citation).
