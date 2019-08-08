#!/bin/bash
# Train LearnToSelectED on ACE 2005 (standard split).
# Reproduces Table 1 of the paper (target: F1 = 74.2).

set -e

python -m eventdetection_gumbel.train --config configs/ace2005.yaml

echo ""
echo "Evaluating on test set..."
python scripts/evaluate.py \
    --checkpoint checkpoints/ace2005/best_model.pt \
    --test_file  data/ace2005/test.json \
    --config     configs/ace2005.yaml
