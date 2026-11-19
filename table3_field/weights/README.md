# Pretrained weights (Table 3, field data)

The weight files are distributed separately (Zenodo, **DOI: 10.5281/zenodo.24340552**).
From the repository root, `python scripts/download_weights.py --table 3` downloads them into this folder
and checks their MD5. On Zenodo the files carry the prefix `field_` (for example `field_multiscale_unet.pth`);
files downloaded by hand can be put in place with `python scripts/download_weights.py --table 3 --from-dir <folder>`.

These are the field-data models, trained on 22,000 synthetic grids (32 x 32 nodes, 400 m spacing; `field_train`
on Zenodo, downloaded by `scripts/download_data.py` from the repository root).
They are a different set from the Table-2 (synthetic data) weights, even though the file names are the same.

Each file is a plain PyTorch `state_dict` with inference weights only (no optimizer state);
`gravdc.load_model(<name>)` builds the matching architecture and loads it. All learned methods use their
raw training weights (no exponential moving average).

| File | `load_model` name | Method (Table 3) | Params | Size | MD5 |
|---|---|---|---|---|---|
| `multiscale_dit.pth` | `multiscale_dit` | Multiscale DiT (ours), 199k updates | 457 M | 1.7 GB | `55433a720f3633f0f606046123917128` |
| `multiscale_unet.pth` | `multiscale_unet` | Multiscale U-Net (ours), 199k updates | 224 M | 854 MB | `7ab2e9b48ef6c68a8e5be31b52f97370` |
| `ddpm_conditional_unet_raw.pth` | `ddpm_unet` | Denoising diffusion (U-Net), raw weights | 222 M | 849 MB | `fc741f22497756375484d32a4023345c` |
| `unet_bansal.pth` | `unet` | Backbone U-Net | 222 M | 849 MB | `393d9f329cf77c554e6eb8fc559baabc` |
| `dcnet.pth` | `dcnet` | DC-Net (Li et al., 2023) | 31 M | 119 MB | `07b5ddd0bb2046b9836afb33eed26afc` |
| `dunet.pth` | `dunet` | DU-Net (Ye et al., 2022) | 5 M | 20 MB | `6f112f6599c5352c7d8edad71cb80f4c` |
| `physics_trained.pth` | `physics_trained` | Physics-trained (Sun et al., 2025) | 222 M | 849 MB | `5552276d5ba88a2196d7acf99f04fd30` |
| `pnp_denoiser.pth` | `pnp_denoiser` | CNN-regularized (PnP denoiser) | 222 M | 849 MB | `f573893c0c725a4317d3be48b7a7e682` |

TTSIDC and TRDC are classical methods and have no weights. Total size about 6 GB.

`physics_trained.pth` has no training set: it is fitted to the 10 test observations through the physics loss
only (the ground truth is never used for fitting). With the field data in `data/test/` (not included),
`python scripts/train_physics_trained.py --out
weights/physics_trained.pth` regenerates it (10,000 updates, seed 0, about 16 minutes on an RTX 3090); on
the tested environment two independent fits gave bit-identical weights and the same MD5 as above.
