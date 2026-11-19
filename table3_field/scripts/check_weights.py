"""Check the Table-3 weight files against their MD5 checksums.  Usage: python scripts/check_weights.py"""
import hashlib
import os
import sys

MD5 = {
    "multiscale_dit.pth": "55433a720f3633f0f606046123917128",
    "multiscale_unet.pth": "7ab2e9b48ef6c68a8e5be31b52f97370",
    "ddpm_conditional_unet_raw.pth": "fc741f22497756375484d32a4023345c",
    "unet_bansal.pth": "393d9f329cf77c554e6eb8fc559baabc",
    "dcnet.pth": "07b5ddd0bb2046b9836afb33eed26afc",
    "dunet.pth": "6f112f6599c5352c7d8edad71cb80f4c",
    "physics_trained.pth": "5552276d5ba88a2196d7acf99f04fd30",
    "pnp_denoiser.pth": "f573893c0c725a4317d3be48b7a7e682",
}
folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
ok = True
for name, ref in MD5.items():
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        print(f"MISSING  {name}")
        ok = False
        continue
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    good = h.hexdigest() == ref
    ok &= good
    print(f"{'OK      ' if good else 'BAD MD5 '} {name}")
sys.exit(0 if ok else 1)
