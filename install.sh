#!/bin/bash
set -euxo pipefail

# Install necessary package for model training
conda install -c conda-forge --yes imbalanced-learn xgboost

echo "Successfully installed imbalanced-learn and xgboost via conda"