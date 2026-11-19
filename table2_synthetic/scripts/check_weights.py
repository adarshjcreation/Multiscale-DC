"""Check the downloaded weight files against their MD5 checksums.  Usage: python scripts/check_weights.py"""
import hashlib
import os
import sys

MD5 = {
    "multiscale_dit.pth": "f9d0fd0f21d0f0486ddf8ef324165c11",
    "multiscale_unet.pth": "360993e45e923cf19c5b85b4bd0b2480",
    "ddpm_conditional_unet_raw.pth": "05ea9d2ed7c191e9586796306fb3c947",
    "unet_bansal.pth": "ac3eb76d9b5325884d942cd68bfa1000",
    "dcnet.pth": "d39b307d2281f1edf34ae910ec34f88c",
    "dunet.pth": "ade0a07a5e73aad8308a788a85f88b7e",
    "physics_trained.pth": "16824a7d8fa347836fc730670725232c",
    "pnp_denoiser.pth": "2a2829481ce66ad87f335d6c303ad730",
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
