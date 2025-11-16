#!/bin/bash

# Exit immediately if a command fails
set -e

# Path to checkpoint inside Modal volume
INIT_CKPT="/data/checkpoints/best_model.pt"

# Path to your training PGN inside the Modal volume
PGN_PATH="/data/train.pgn"

# Default hyperparameters (edit if needed)
EPOCHS=10
BATCH_SIZE=128
LR=3e-4
NUM_WORKERS=4
CACHE_PATH="/data/pgn_cache.pt"
CHECKPOINT_DIR="/data/checkpoints"

echo "=========================================="
echo "   Running Modal Supervised Training"
echo "=========================================="
echo "PGN:            $PGN_PATH"
echo "Init checkpoint: $INIT_CKPT"
echo "Epochs:          $EPOCHS"
echo "Batch size:      $BATCH_SIZE"
echo "LR:              $LR"
echo "------------------------------------------"

modal run src/training/train_cli.py \
  --use-modal \
  --pgn "$PGN_PATH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --num-workers "$NUM_WORKERS" \
  --cache-path "$CACHE_PATH" \
  --checkpoint-dir "$CHECKPOINT_DIR"

echo "=========================================="
echo "   Training Complete"
echo "=========================================="
