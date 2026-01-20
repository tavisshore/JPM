#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere, but resolve paths relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
Q3_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"          # scripts/question_3
REPO_ROOT="$(cd "${Q3_DIR}/../.." && pwd)"        # JPM/

ENV_FILE="${Q3_DIR}/environment.yml"
ENV_NAME="choice-learn-lu25"

echo "Repo root: ${REPO_ROOT}"
echo "Using environment file: ${ENV_FILE}"
echo "Environment name: ${ENV_NAME}"
echo ""

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda/Miniforge and retry."
  echo "Fallback (pip):"
  echo "  pip install -r ${Q3_DIR}/env/requirements-choicelearn.txt"
  exit 1
fi

# Create env if missing
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
else
  echo "Creating conda env '${ENV_NAME}'..."
  conda env create -f "${ENV_FILE}"
fi

echo ""
echo "Next steps:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "IMPORTANT (required for this repo's code):"
echo "  export TF_USE_LEGACY_KERAS=1"
echo ""
echo "Smoke test:"
echo "  python scripts/question_3/part_1/reproduce_table1.py"
