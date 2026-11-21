#!/usr/bin/env bash
# Reproduce Table 2 and Table 3: run every notebook of both folders and assemble the tables.
# Table 3 runs only if the field test data are in table3_field/data/test
# (download them from Zenodo with: python scripts/download_data.py field_test).
# Usage: bash scripts/run_all.sh            (uses the Jupyter kernel "python3")
#        KERNEL=myenv bash scripts/run_all.sh
# About 5 minutes (Table 2) + 8.5 minutes (Table 3) on an RTX 3090.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "=== table2_synthetic"
bash "$ROOT/table2_synthetic/scripts/run_all.sh"
echo "=== table3_field"
if python "$ROOT/table3_field/scripts/check_data.py" > /dev/null; then
  bash "$ROOT/table3_field/scripts/run_all.sh"
  T3="table3_field/results/table3.md"
else
  echo "skipped: the field test data are missing; download them with: python scripts/download_data.py field_test"
  T3="not run (stored results: table3_field/results/table3.md)"
fi
echo
echo "Table 2: table2_synthetic/results/table2.md"
echo "Table 3: $T3"
