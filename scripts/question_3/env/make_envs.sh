#!/usr/bin/env bash
set -e

echo "Creating standalone env: lu25_standalone"
conda create -n lu25_standalone python=3.11 -y
conda run -n lu25_standalone python -m pip install --upgrade pip
conda run -n lu25_standalone pip install -r env/requirements-standalone.txt

echo "Creating choice-learn env: lu25_choicelearn"
conda create -n lu25_choicelearn python=3.11 -y
conda run -n lu25_choicelearn python -m pip install --upgrade pip
conda run -n lu25_choicelearn pip install -r env/requirements-choicelearn.txt

echo
echo "Standalone:     conda activate lu25_standalone"
echo "Choice-learn:   conda activate lu25_choicelearn"
echo "IMPORTANT for choice-learn runs: TF_USE_LEGACY_KERAS=1"
