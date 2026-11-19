"""Download the pretrained weights from Zenodo into table2_synthetic/weights/ and table3_field/weights/.

The Zenodo record (DOI 10.5281/zenodo.24340552) holds 16 weight files: 8 for Table 2 (prefix ``synthetic_``) and
8 for Table 3 (prefix ``field_``), plus the datasets (see scripts/download_data.py).
This script saves each one under its plain name in the matching ``weights/`` folder (for example
``synthetic_multiscale_unet.pth`` -> ``table2_synthetic/weights/multiscale_unet.pth``) and checks its MD5.
Files that are already present with the correct MD5 are skipped. Total size about 12 GB (6 GB per table).

Usage:
    python scripts/download_weights.py                  # both tables
    python scripts/download_weights.py --table 2        # Table 2 only (or --table 3)
    python scripts/download_weights.py --from-dir DIR   # use files already downloaded by hand into DIR
"""
import argparse
import hashlib
import os
import shutil
import sys
import urllib.error
import urllib.request

ZENODO_RECORD = "24340552"  # Zenodo record number (the digits at the end of the DOI 10.5281/zenodo.24340552)
ZENODO_URL = "https://zenodo.org"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Zenodo file name -> (table, MD5); the local name is the Zenodo name without its prefix
FILES = {
    "synthetic_multiscale_dit.pth": (2, "f9d0fd0f21d0f0486ddf8ef324165c11"),
    "synthetic_multiscale_unet.pth": (2, "360993e45e923cf19c5b85b4bd0b2480"),
    "synthetic_ddpm_conditional_unet_raw.pth": (2, "05ea9d2ed7c191e9586796306fb3c947"),
    "synthetic_unet_bansal.pth": (2, "ac3eb76d9b5325884d942cd68bfa1000"),
    "synthetic_dcnet.pth": (2, "d39b307d2281f1edf34ae910ec34f88c"),
    "synthetic_dunet.pth": (2, "ade0a07a5e73aad8308a788a85f88b7e"),
    "synthetic_physics_trained.pth": (2, "16824a7d8fa347836fc730670725232c"),
    "synthetic_pnp_denoiser.pth": (2, "2a2829481ce66ad87f335d6c303ad730"),
    "field_multiscale_dit.pth": (3, "55433a720f3633f0f606046123917128"),
    "field_multiscale_unet.pth": (3, "7ab2e9b48ef6c68a8e5be31b52f97370"),
    "field_ddpm_conditional_unet_raw.pth": (3, "fc741f22497756375484d32a4023345c"),
    "field_unet_bansal.pth": (3, "393d9f329cf77c554e6eb8fc559baabc"),
    "field_dcnet.pth": (3, "07b5ddd0bb2046b9836afb33eed26afc"),
    "field_dunet.pth": (3, "6f112f6599c5352c7d8edad71cb80f4c"),
    "field_physics_trained.pth": (3, "5552276d5ba88a2196d7acf99f04fd30"),
    "field_pnp_denoiser.pth": (3, "f573893c0c725a4317d3be48b7a7e682"),
}
TABLE_DIR = {2: "table2_synthetic", 3: "table3_field"}
PREFIX = {2: "synthetic_", 3: "field_"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    """Download ``url`` to ``dest`` (via ``dest.part``), printing the progress."""
    part = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "gravdc-download-weights"})
    with urllib.request.urlopen(req) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 2**20:8.0f} / {total / 2**20:.0f} MB", end="", flush=True)
    print()
    os.replace(part, dest)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", type=int, choices=(2, 3), action="append",
                   help="table whose weights to fetch (default: both; can be repeated)")
    p.add_argument("--record", default=ZENODO_RECORD, help="Zenodo record number (default: %(default)s)")
    p.add_argument("--base-url", default=ZENODO_URL, help="Zenodo server (default: %(default)s)")
    p.add_argument("--from-dir", help="folder that already contains the Zenodo files (copied, not downloaded)")
    p.add_argument("--link", action="store_true", help="with --from-dir: hard-link instead of copying")
    args = p.parse_args()
    tables = sorted(set(args.table or (2, 3)))

    if not args.from_dir and not args.record.isdigit():
        sys.exit("The Zenodo record number is not set: pass --record NUMBER (or edit ZENODO_RECORD).")

    ok = True
    for zname, (table, ref) in FILES.items():
        if table not in tables:
            continue
        dest = os.path.join(ROOT, TABLE_DIR[table], "weights", zname[len(PREFIX[table]):])
        rel = os.path.relpath(dest, ROOT)
        if os.path.exists(dest) and md5(dest) == ref:
            print(f"OK (present)  {rel}")
            continue
        if args.from_dir:
            src = os.path.realpath(os.path.join(args.from_dir, zname))
            if not os.path.exists(src):
                print(f"MISSING       {src}")
                ok = False
                continue
            if os.path.exists(dest):
                os.remove(dest)
            try:
                (os.link if args.link else shutil.copyfile)(src, dest)
            except OSError:  # hard links do not work across file systems: copy instead
                shutil.copyfile(src, dest)
        else:
            url = f"{args.base_url.rstrip('/')}/records/{args.record}/files/{zname}?download=1"
            print(f"downloading   {zname}")
            try:
                download(url, dest)
            except (urllib.error.URLError, OSError) as e:
                print(f"FAILED        {zname}: {e}")
                ok = False
                continue
        good = md5(dest) == ref
        ok &= good
        print(f"{'OK           ' if good else 'BAD MD5      '} {rel}")
    if not ok:
        sys.exit("Some weight files are missing or corrupt (see above).")
    print("All weight files present and verified.")


if __name__ == "__main__":
    main()
