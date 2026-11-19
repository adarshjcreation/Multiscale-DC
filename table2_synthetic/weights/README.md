# Pretrained weights

The weight files are distributed separately (Zenodo, **DOI: 10.5281/zenodo.24340552**).
From the repository root, `python scripts/download_weights.py --table 2` downloads them into this folder
and checks their MD5. On Zenodo the files carry the prefix `synthetic_` (for example `synthetic_multiscale_unet.pth`);
files downloaded by hand can be put in place with `python scripts/download_weights.py --table 2 --from-dir <folder>`.

Each file is a plain PyTorch `state_dict` with inference weights only (no optimizer state);
`gravdc.load_model(<name>)` builds the matching architecture and loads it. All learned methods use their
raw training weights (no exponential moving average).

Except Physics-Trained (fitted to the test observations), the networks were trained on the 22,000 synthetic grids
(32 x 32 nodes, 50 m spacing) of `synthetic_train` on Zenodo, downloaded by `scripts/download_data.py` from the
repository root.

| File | `load_model` name | Method (Table 2) | Params | Size | MD5 |
|---|---|---|---|---|---|
| `multiscale_dit.pth` | `multiscale_dit` | Multiscale DiT (ours) | 457 M | 1.7 GB | `f9d0fd0f21d0f0486ddf8ef324165c11` |
| `multiscale_unet.pth` | `multiscale_unet` | Multiscale U-Net (ours) | 224 M | 854 MB | `360993e45e923cf19c5b85b4bd0b2480` |
| `ddpm_conditional_unet_raw.pth` | `ddpm_unet` | Denoising diffusion (U-Net), raw weights | 222 M | 849 MB | `05ea9d2ed7c191e9586796306fb3c947` |
| `unet_bansal.pth` | `unet` | Backbone U-Net | 222 M | 849 MB | `ac3eb76d9b5325884d942cd68bfa1000` |
| `dcnet.pth` | `dcnet` | DC-Net (Li et al., 2023) | 31 M | 119 MB | `d39b307d2281f1edf34ae910ec34f88c` |
| `dunet.pth` | `dunet` | DU-Net (Ye et al., 2022) | 5 M | 20 MB | `ade0a07a5e73aad8308a788a85f88b7e` |
| `physics_trained.pth` | `physics_trained` | Physics-trained (Sun et al., 2025) | 222 M | 849 MB | `16824a7d8fa347836fc730670725232c` |
| `pnp_denoiser.pth` | `pnp_denoiser` | CNN-regularized (PnP denoiser) | 222 M | 849 MB | `2a2829481ce66ad87f335d6c303ad730` |

TTSIDC and TRDC are classical methods and have no weights. Total size about 6 GB.
