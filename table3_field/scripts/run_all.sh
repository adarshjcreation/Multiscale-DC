#!/usr/bin/env bash
# Run every notebook in order (01-10, then 00) and save the executed notebooks in place.
# Usage: bash scripts/run_all.sh            (uses the Jupyter kernel "python3")
#        KERNEL=myenv bash scripts/run_all.sh
set -euo pipefail
# the field data are on Zenodo, not in git: python scripts/download_data.py field_test (see data/README.md)
if ! python "$(dirname "$0")/check_data.py" > /dev/null; then
  echo "Table 3 needs the 10 field test patches in data/test: run python scripts/download_data.py field_test" >&2
  echo "in the top folder of the repository (see data/README.md)." >&2
  python "$(dirname "$0")/check_data.py" >&2 || true
  exit 1
fi
cd "$(dirname "$0")/../notebooks"
KERNEL="${KERNEL:-python3}"
for nb in 01_TTSIDC 02_TRDC 03_DU-Net 04_DC-Net 05_U-Net 06_Physics-Trained 07_CNN-Regularized \
          08_Denoising_Diffusion 09_Multiscale_DiT 10_Multiscale_UNet 00_Reproduce_Table3; do
  start=$(date +%s)
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name="$KERNEL" "$nb.ipynb" 2>/dev/null
  echo "$nb  done in $(( $(date +%s) - start )) s"
done
echo "Table 3 written to results/table3.md and results/table3.csv"
