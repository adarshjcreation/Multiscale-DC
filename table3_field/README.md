# Multiscale Learning for Downward Continuation of Gravity Data: Table 3 (field data)

This folder reproduces **Table 3** of the paper, the comparison of downward-continuation approaches on
field gravity data. It contains the code, the pretrained weights (downloaded separately), one notebook per
method and the per-patch metrics of every row. Running the notebooks regenerates every entry of the table and
checks it against the reference values.

**The field data are not included.** The 10 test patches were provided by the Geological Survey of India
and cannot be redistributed; see `data/README.md` for the expected files and their MD5 checksums. With the
data in `data/test/`, the table reproduces exactly. The executed notebooks in this repository were run with
the data, but their figures of the data are switched off (`SHOW_FIGURE = False` in the last cell).

## Contents

```
data/test/                10 field test patches (.mat, variable `d`, 32 x 32 nodes, 400 m spacing);
                          not included, see data/README.md
gravdc/                   Python package shared by the notebooks
  data.py                   loading + per-patch scaling to [-1, 1]
  physics.py                upward-continuation operators (single 800 m step; multiscale 40 x 20 m)
  metrics.py                PSNR, SSIM, RE, RMSE on the central 24 x 24 window
  classical.py              TTSIDC and Tikhonov (TRDC) with automatic parameter selection
  sampling.py               multiscale DC sampling, DDPM sampler, plug-and-play ADMM
  models/                   U-Net (ConvNeXt), DiT-L/2, DC-Net, DU-Net
  weights.py                model constructors and weight files
  precision.py              full-FP32, deterministic evaluation (applied on import)
  table3.py                 reference values of Table 3
  plotting.py               quick-look figures
notebooks/
  01_TTSIDC                 02_TRDC (TRDC and Oracle TRDC)      03_DU-Net
  04_DC-Net                 05_U-Net                            06_Physics-Trained
  07_CNN-Regularized        08_Denoising_Diffusion              09_Multiscale_DiT
  10_Multiscale_UNet        00_Reproduce_Table3 (collects and checks all rows)
weights/                  pretrained weights (see weights/README.md)
results/                  one JSON file per row (per-patch metrics) and the assembled table3.md / table3.csv
scripts/run_all.sh        runs all notebooks in order
scripts/check_weights.py  checks the weight files (MD5)
scripts/check_data.py     checks the field test patches in data/test (MD5)
scripts/train_physics_trained.py   regenerates weights/physics_trained.pth (fit on the test observations)
```

## Setup

```bash
pip install -r ../requirements.txt                 # exact package versions (repository root)
python ../scripts/download_weights.py --table 3   # weights from Zenodo into weights/, MD5-checked
python scripts/check_weights.py                    # optional: check the weight files again
# copy the 10 field test patches into data/test/ (not included, see data/README.md), then
python scripts/check_data.py
```

A CUDA GPU is recommended. Everything also runs on CPU (see Reproducibility).

## Running

```bash
bash scripts/run_all.sh        # or run notebooks 01 ... 10 and then 00 by hand
```

On an RTX 3090 every notebook takes less than 15 s, except `08_Denoising_Diffusion` (about 7 minutes:
3 seeds x 10 samples x 500 reverse steps); the whole set takes about 8.5 minutes. Each notebook prints its
result next to the reference value. Notebook `00` assembles the table, marks every row with ✓ when all four
metrics (mean and standard deviation) equal Table 3 to 4 decimals, and writes `results/table3.md` and
`results/table3.csv`.

## Problem setting and evaluation protocol

* **Ground truth** $x_0$: 10 field gravity patches of 32 x 32 nodes (400 m spacing), each scaled to [-1, 1].
  The networks were trained on 22,000 synthetic grids with the same 400 m spacing (`field_train` on Zenodo,
  see `../scripts/download_data.py`); no field data were
  used for training (Physics-Trained is fitted to the test observations, see Methods).
* **Observation**: $x_0$ continued 800 m upward.
  * Multiscale models, TTSIDC and TRDC: the multiscale operator, 40 continuations of 20 m, each with zero
    padding to 36 x 36 and cropping back to 32 x 32.
  * DU-Net, DC-Net, U-Net, Physics-trained and CNN-regularized: a single 800 m continuation with zero
    padding to 36 x 36.
  * Denoising diffusion: the same single 800 m continuation, with zero padding to 64 x 64 as in its training.
* **Metrics**: PSNR (with the data range of the ground truth), SSIM, relative error
  RE = ‖x̂ − x‖ / ‖x‖ and RMSE, computed on the central 24 x 24 window (the outer 4 nodes are affected by
  the FFT edge effects). Every row is the mean ± standard deviation (ddof = 0) over all 10 patches.
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
| Physics-Trained (Sun et al., 2025) | 06 | `physics_trained.pth` | fitted directly on the 10 test observations through the physics loss ‖Ax̂ − y‖₁ (10,000 updates, seed 0); the ground truth is used only for evaluation |
| CNN-regularized | 07 | `pnp_denoiser.pth` | plug-and-play ADMM, 100 iterations, λ = 2 |
| Denoising diffusion (U-Net) | 08 | `ddpm_conditional_unet_raw.pth` | 500 reverse steps (chosen on training patches), metrics on the posterior mean of 10 samples per patch, seed 1 |
| Multiscale DiT (ours) | 09 | `multiscale_dit.pth` | Single-shot and Stabilized DC |
| Multiscale U-Net (ours) | 10 | `multiscale_unet.pth` | Single-shot, Stabilized DC and Recursive DC |

All learned methods are evaluated with their raw training weights (no exponential moving average).

## Table 3

Output of `00_Reproduce_Table3.ipynb` (`results/table3.md`). Bold: best value of each column.

| Approach | PSNR (dB) ↑ | SSIM ↑ | RE ↓ | RMSE ↓ |
|---|---|---|---|---|
| TTSIDC | 36.8247 ± 2.5611 | 0.9882 ± 0.0076 | 0.0746 ± 0.0243 | 0.0262 ± 0.0097 |
| Oracle TRDC | 40.9487 ± 2.7437 | 0.9910 ± 0.0051 | 0.0455 ± 0.0125 | 0.0160 ± 0.0055 |
| TRDC | 37.7190 ± 2.1283 | 0.9862 ± 0.0084 | 0.0656 ± 0.0176 | 0.0231 ± 0.0080 |
| DU-Net | 36.2222 ± 1.8690 | 0.9861 ± 0.0088 | 0.0768 ± 0.0144 | 0.0270 ± 0.0073 |
| DC-Net | 33.9575 ± 1.9905 | 0.9771 ± 0.0116 | 0.1003 ± 0.0208 | 0.0353 ± 0.0101 |
| Physics-Trained | 26.4089 ± 1.7020 | 0.8490 ± 0.1073 | 0.2391 ± 0.0506 | 0.0816 ± 0.0128 |
| CNN-regularized | 28.3958 ± 1.7156 | 0.9254 ± 0.0316 | 0.1888 ± 0.0293 | 0.0663 ± 0.0158 |
| U-Net | 32.4771 ± 1.9784 | 0.9724 ± 0.0142 | 0.1201 ± 0.0277 | 0.0421 ± 0.0119 |
| Denoising diffusion (U-Net) | 38.4493 ± 2.7439 | 0.9911 ± 0.0089 | 0.0618 ± 0.0214 | 0.0218 ± 0.0091 |
| Multiscale DiT (Single-shot) | 39.8545 ± 2.6601 | 0.9940 ± 0.0067 | 0.0525 ± 0.0181 | 0.0186 ± 0.0078 |
| Multiscale DiT (Stabilized DC) | 39.3045 ± 2.6594 | 0.9933 ± 0.0073 | 0.0561 ± 0.0197 | 0.0198 ± 0.0084 |
| Multiscale U-Net (Single-shot) | **41.8093 ± 3.0735** | **0.9953 ± 0.0069** | **0.0427 ± 0.0186** | **0.0152 ± 0.0078** |
| Multiscale U-Net (Stabilized DC) | 40.8483 ± 3.0420 | 0.9943 ± 0.0078 | 0.0478 ± 0.0203 | 0.0169 ± 0.0085 |
| Multiscale U-Net (Recursive DC) | 28.6774 ± 2.1266 | 0.9499 ± 0.0158 | 0.1836 ± 0.0362 | 0.0641 ± 0.0156 |

**Denoising diffusion: seeds and number of reverse steps.** The sampler is stochastic, so its row depends
on the seed. Notebook 08 runs seeds 0, 1 and 2 and reports seed 1:

| Seed | PSNR (dB) | SSIM | RE | RMSE |
|---|---|---|---|---|
| 0 | 38.4442 ± 2.7472 | 0.9910 ± 0.0089 | 0.0619 ± 0.0214 | 0.0219 ± 0.0091 |
| 1 (Table 3) | 38.4493 ± 2.7439 | 0.9911 ± 0.0089 | 0.0618 ± 0.0214 | 0.0218 ± 0.0091 |
| 2 | 38.4391 ± 2.7433 | 0.9910 ± 0.0089 | 0.0619 ± 0.0214 | 0.0219 ± 0.0091 |

The sampler starts from Gaussian noise and runs the last `steps` steps of the 1000-step training schedule.
The number of steps was chosen on 20 training patches (every 1100th file of the training set, seed 0):
the posterior mean reaches 56.19, 56.36, 56.66 and 56.29 dB with 100, 250, 500 and 1000 steps, so 500
steps are used. On the test patches the row hardly depends on the step count (posterior mean, seeds 0–2:
38.43–38.44 dB with 100 steps, 38.44–38.45 dB with 250 steps, 38.44–38.45 dB with 500 steps).

**Physics-Trained: initialisation.** The network is fitted to the test observations from a random
initialisation, so the row depends on the seed. `scripts/train_physics_trained.py` (10,000 full-batch
updates, Adam, learning rate 2e-5, L1 physics loss) gives:

| Seed | PSNR (dB) | SSIM | RE | RMSE |
|---|---|---|---|---|
| 0 (Table 3, script default) | 26.4089 ± 1.7020 | 0.8490 ± 0.1073 | 0.2391 ± 0.0506 | 0.0816 ± 0.0128 |
| 1 | 27.4952 ± 1.6343 | 0.8839 ± 0.0753 | 0.2094 ± 0.0362 | 0.0717 ± 0.0089 |
| 2 | 25.5466 ± 3.3218 | 0.8243 ± 0.1057 | 0.2726 ± 0.0897 | 0.0954 ± 0.0390 |

The number of updates and the seed were fixed without the ground truth. The original fit
(Physics_trained_net.ipynb) was continued over several sessions, so its total length is not recorded;
after 10,000 updates the physics loss ‖Ax̂ − y‖₁ is about 1e-3, the level the original fit reached.

## Differences from the submitted Table 3

The submitted Table 3 averaged over 8 of these 10 patches (data18 and data30 were not included; the other
8 files are identical) and ran the networks in TF32. The table below compares the PSNR of the submitted
table with this release restricted to the same 8 patches (the per-patch results of this release averaged
over those 8 patches), and with the full release (10 patches).

| Approach | Submitted (8 patches) | This release, same 8 patches | This release, 10 patches | Reason for the difference on the same 8 patches |
|---|---|---|---|---|
| TTSIDC | 36.9855 | 36.9041 | 36.8247 | the submitted value used 13 iterations for data14 and data32, set by hand; here the iterations are selected automatically for every patch (10 everywhere) |
| Oracle TRDC | 40.1561 | 40.1561 | 40.9487 | – |
| TRDC | 37.4237 | 37.4237 | 37.7190 | – |
| DU-Net | 36.2475 | 36.2477 | 36.2222 | FP32 instead of TF32 |
| DC-Net | 34.1444 | 34.1443 | 33.9575 | FP32; the submitted RMSE 0.0392 ± 0.0093 was a transcription error (the notebook printed 0.0350 ± 0.0103) |
| Physics-Trained | 26.6088 | 26.5720 | 26.4089 | refitted on all 10 observations (the submitted weights were fitted on the 8 patches) |
| CNN-regularized | 28.5356 | 28.5337 | 28.3958 | FP32 |
| U-Net | 32.6414 | 32.6417 | 32.4771 | FP32 |
| Denoising diffusion (U-Net) | 38.7687 | 38.7210 | 38.4493 | raw weights instead of EMA; metrics on the posterior mean instead of the average of the per-sample metrics; different seed |
| Multiscale DiT (Single-shot) | 40.0278 | 40.0278 | 39.8545 | – |
| Multiscale DiT (Stabilized DC) | 39.4714 | 39.4713 | 39.3045 | FP32 |
| Multiscale U-Net (Single-shot) | 41.9882 | 42.0083 | 41.8093 | FP32 |
| Multiscale U-Net (Stabilized DC) | 41.0285 | 41.0285 | 40.8483 | – |
| Multiscale U-Net (Recursive DC) | 29.0709 | 29.1766 | 28.6774 | FP32 (the 40-step recursion amplifies TF32 rounding) |

With the two added patches the learned methods drop by 0.03–0.27 dB (Multiscale U-Net Recursive DC by
0.50 dB), mainly because they score data30 below their 8-patch average. Oracle TRDC (+0.79 dB) and TRDC
(+0.30 dB) rise, because both new patches are easy for them. The best approach is unchanged: Multiscale
U-Net (Single-shot) is the best in all four metrics. Among the other rows, Oracle TRDC (which uses the
ground truth) moves from below to just above Multiscale U-Net (Stabilized DC), 40.95 vs 40.85 dB.

## Reproducibility

* **Precision.** On Ampere and newer GPUs PyTorch runs float32 convolutions in TF32 by default, which
  changes the 3rd–4th decimal of the PSNR (up to 0.1 dB for Multiscale U-Net Recursive DC on these
  patches) and makes it depend on the GPU and the cuDNN algorithm. `gravdc` therefore switches to full
  FP32 with deterministic cuDNN when it is imported (`gravdc/precision.py`).
* **Random numbers.** The only stochastic method, denoising diffusion, draws all its noise from a CPU
  random generator with a fixed seed, so the noise is the same on every device. The Physics-Trained fit
  uses a fixed seed and deterministic cuDNN.
* **Checks that were run:**
  * Four complete runs from an empty `results/` folder (RTX 3090), one notebook by notebook and three with
    `scripts/run_all.sh`, gave bit-identical per-patch metrics for all 14 rows and byte-identical
    `table3.md` / `table3.csv`, and all 14 rows match `gravdc/table3.py`.
  * Physics-Trained: three independent fits with seed 0 (10,000 updates each) gave bit-identical weights;
    the training curves agree at every logged update.
  * A CPU-only run of notebooks 01–07, 09 and 10: TTSIDC, TRDC and Oracle TRDC are identical to the GPU
    run (they run on the CPU anyway); DU-Net, DC-Net, U-Net, Physics-Trained and CNN-regularized agree
    with Table 3 to 4 decimals; the multiscale rows differ by at most 0.0001 dB in the PSNR, except
    Multiscale U-Net Recursive DC (28.6760 vs 28.6774 dB), and in the last digit of a few standard
    deviations. These are floating-point rounding differences between CPU and GPU kernels, which the
    40-step multiscale routines accumulate (most of all the recursive one).
  * Denoising diffusion: a repeated GPU run of all three seeds reproduced the stored per-patch metrics
    exactly. Seed 1 (the Table-3 row) on CPU, 10 patches x 10 samples x 500 steps: the samples agree with
    the GPU samples to 2e-5 and the per-patch PSNR to 0.0001 dB, and the row equals Table 3 to 4 decimals
    in all four metrics (38.4493 ± 2.7439 dB). One seed takes about 35 minutes on a 16-core CPU, so the full
    notebook 08 takes about 1.5–2 hours on CPU.
* **Tested environment:** Python 3.11.6, PyTorch 2.9.0 with CUDA 12.8 and cuDNN 9.10, NVIDIA GeForce
  RTX 3090, Linux. The exact package versions are pinned in `../requirements.txt`.

## Citation

See the [repository README](../README.md#citation).
