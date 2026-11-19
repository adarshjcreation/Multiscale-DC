"""Model constructors and their released weight files (see README for the Zenodo download)."""
import os

import torch

from .models.dcnet import DCNet
from .models.dit import DiT_L_2
from .models.dunet import DUnet
from .models.unet import ConditionalUnet, DownContinuationNet, Unet

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")

MODELS = {
    # name: (constructor, weight file)
    "multiscale_dit": (lambda: DiT_L_2(in_channels=1, input_size=32), "multiscale_dit.pth"),
    "multiscale_unet": (lambda: Unet(dim=128, dim_mults=(1, 2, 4, 8), channels=1), "multiscale_unet.pth"),
    # raw (non-EMA) training weights, like every other learned method in Table 2
    "ddpm_unet": (lambda: ConditionalUnet(dim=128, dim_mults=(1, 2, 4, 8), channels=1, cond_channels=1, out_dim=1,
                                          with_time_emb=False), "ddpm_conditional_unet_raw.pth"),
    "unet": (lambda: Unet(dim=128, dim_mults=(1, 2, 4, 8), channels=1, with_time_emb=False), "unet_bansal.pth"),
    "dcnet": (lambda: DCNet(in_channels=1, out_channels=1), "dcnet.pth"),
    "dunet": (lambda: DUnet(input_channels=1), "dunet.pth"),
    "physics_trained": (DownContinuationNet, "physics_trained.pth"),
    "pnp_denoiser": (lambda: Unet(dim=128, dim_mults=(1, 2, 4, 8), channels=1, with_time_emb=False),
                     "pnp_denoiser.pth"),
}


def load_model(name, device="cuda", weights_dir=WEIGHTS_DIR):
    """Build model ``name`` (a key of ``MODELS``), load its released weights and put it in eval mode."""
    build, fname = MODELS[name]
    model = build()
    state = torch.load(os.path.join(weights_dir, fname), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
