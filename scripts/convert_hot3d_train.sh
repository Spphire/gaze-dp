#!/bin/bash
cd /mnt/workspace/shenyibo/gaze-wam
export PYTHONPATH=/mnt/workspace/shenyibo/gaze-wam:$PYTHONPATH

echo "=== Converting HOT3D train split to zarr ==="
.venv/bin/python scripts/convert_hot3d_processed_to_open_zarr.py \
  --processed-root /mnt/workspace/shenyibo/datasets/HOT3D/processed \
  --output-zarr data/hot3d_open_train.zarr \
  --sequence-file data/hot3d_train_sequences.txt \
  --image-size 256 256 \
  --stride 1 \
  --heatmap-storage token \
  --heatmap-token-grid 16 16 \
  --overwrite

echo "=== Train zarr complete ==="
ls -lh data/hot3d_open_train.zarr
du -sh data/hot3d_open_train.zarr
