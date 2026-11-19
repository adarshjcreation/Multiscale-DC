"""Download the synthetic datasets from Zenodo (DOI 10.5281/zenodo.24340552) into datasets/.

The datasets are not needed to reproduce Tables 2 and 3 (the 10 synthetic test grids are also in
table2_synthetic/data/test). Each archive is checked against its MD5, unpacked into datasets/<name>/ and deleted.

    synthetic_train   22,000 grids, 50 m spacing: training set of the Table-2 networks          (177 MB)
    synthetic_test    the 10 synthetic test grids of Table 2                                      (0.1 MB)
    field_train       22,000 synthetic grids, 400 m spacing: training set of the Table-3 networks (182 MB)

The field test patches of Table 3 are not on Zenodo: they cannot be redistributed (see table3_field/data/README.md).

Usage:
    python scripts/download_data.py                          # all three datasets
    python scripts/download_data.py synthetic_train          # only the named ones
    python scripts/download_data.py --from-dir DIR           # use archives already downloaded by hand into DIR
    python scripts/download_data.py --keep-zip               # keep the archives in datasets/
"""
import argparse
import os
import shutil
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_weights import ZENODO_RECORD, ZENODO_URL, download, md5  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = {
    # name: (number of .mat files, MD5 of <name>.zip)
    "synthetic_train": (22000, "9bd4d5e7cca2210a7a11f53c42ff2736"),
    "synthetic_test": (10, "422aeaf7954ac5c31928a9314ed4ed06"),
    "field_train": (22000, "e43cde311f9eacb2e8313af8634764e5"),
}


def count_mat(folder):
    return sum(f.endswith(".mat") for f in os.listdir(folder)) if os.path.isdir(folder) else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", metavar="name", help=f"datasets to fetch: {', '.join(DATASETS)} (default: all)")
    p.add_argument("--out", default=os.path.join(ROOT, "datasets"), help="target folder (default: %(default)s)")
    p.add_argument("--record", default=ZENODO_RECORD, help="Zenodo record number (default: %(default)s)")
    p.add_argument("--base-url", default=ZENODO_URL, help="Zenodo server (default: %(default)s)")
    p.add_argument("--from-dir", help="folder that already contains the archives (not downloaded)")
    p.add_argument("--keep-zip", action="store_true", help="keep the downloaded archives")
    args = p.parse_args()
    unknown = [n for n in args.names if n not in DATASETS]
    if unknown:
        p.error(f"unknown dataset(s): {', '.join(unknown)}; choose from {', '.join(DATASETS)}")
    os.makedirs(args.out, exist_ok=True)

    ok = True
    for name in args.names or list(DATASETS):
        n_files, ref = DATASETS[name]
        target = os.path.join(args.out, name)
        if count_mat(target) == n_files:
            print(f"OK (present)  {name}: {n_files} files")
            continue
        if args.from_dir:
            archive = os.path.join(args.from_dir, name + ".zip")
            if not os.path.exists(archive):
                print(f"MISSING       {archive}")
                ok = False
                continue
        else:
            archive = os.path.join(args.out, name + ".zip")
            url = f"{args.base_url.rstrip('/')}/records/{args.record}/files/{name}.zip?download=1"
            print(f"downloading   {name}.zip")
            try:
                download(url, archive)
            except OSError as e:  # includes urllib.error.URLError
                print(f"FAILED        {name}.zip: {e}")
                ok = False
                continue
        if md5(archive) != ref:
            print(f"BAD MD5       {archive}")
            ok = False
            continue
        if os.path.isdir(target):
            shutil.rmtree(target)
        with zipfile.ZipFile(archive) as z:  # members are <name>/<file>.mat; CRC-checked on extraction
            z.extractall(args.out)
        if not args.from_dir and not args.keep_zip:
            os.remove(archive)
        good = count_mat(target) == n_files
        ok &= good
        print(f"{'OK           ' if good else 'INCOMPLETE   '} {name}: {count_mat(target)} files in "
              f"{os.path.relpath(target, ROOT)}")
    if not ok:
        sys.exit("Some datasets are missing or corrupt (see above).")


if __name__ == "__main__":
    main()
