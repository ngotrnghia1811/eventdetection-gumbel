#!/bin/bash
# Train LearnToSelectED on TAC KBP 2015 (standard split).
# Reproduces Table 2 of the paper (target: F1 = 61.1).

set -e

python -m eventdetection_gumbel.train --config configs/tac2015.yaml

echo ""
echo "Evaluating on test set..."
python scripts/evaluate.py \
    --checkpoint checkpoints/tac2015/best_model.pt \
    --test_file  data/tac2015/test.json \
    --config     configs/tac2015.yaml
