#!/usr/bin/env bash
# Run every notebook in order (01-10, then 00) and save the executed notebooks in place.
# Usage: bash scripts/run_all.sh            (uses the Jupyter kernel "python3")
#        KERNEL=myenv bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/../notebooks"
KERNEL="${KERNEL:-python3}"
for nb in 01_TTSIDC 02_TRDC 03_DU-Net 04_DC-Net 05_U-Net 06_Physics-Trained 07_CNN-Regularized \
          08_Denoising_Diffusion 09_Multiscale_DiT 10_Multiscale_UNet 00_Reproduce_Table2; do
  start=$(date +%s)
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name="$KERNEL" "$nb.ipynb" 2>/dev/null
  echo "$nb  done in $(( $(date +%s) - start )) s"
done
echo "Table 2 written to results/table2.md and results/table2.csv"
