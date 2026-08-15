# Deterministic CNN baseline

This implementation uses a compact U-Net to predict normalized CIELAB chrominance from luminance:

```text
L [B, 1, 32, 32] -> CNN -> ab [B, 2, 32, 32]
```

The predicted `ab` channels are combined with the input `L` channel only for RGB reconstruction. This is a continuous regression baseline, not the 313-bin classification approach.

## Data

The shared preprocessing pipeline provides fixed CIFAR-10 splits of 45,000 training, 5,000 validation, and 10,000 test images. Each HDF5 file contains:

| Key | Shape | Stored dtype |
|---|---|---|
| `L` | `[N, 1, 32, 32]` | `float16` |
| `ab` | `[N, 2, 32, 32]` | `float16` |
| `labels` | `[N]` | integer |

The stored normalization is:

```text
L_normalized = L / 50 - 1
ab_normalized = ab / 128
```

`dataset.py` converts `L` and `ab` to `float32` without further preprocessing. The CNN experiments use the existing HDF5 files and do not rerun `preprocess_image_colorization_data.py`.

## Setup

Run all commands from the repository root:

```bash
python3 -m pip install -r CNN/requirements.txt
python3 CNN/download_data.py --output-dir shared_data
```

The downloader uses the project's shared Google Drive folder by default. Check the downloaded data and model interface with:

```bash
find shared_data -name "*.h5"
python3 CNN/dataset.py shared_data/cifar10_lab_v1/train_lab.h5
python3 CNN/model.py
```

The model check should report output shape `(4, 2, 32, 32)` and 1,947,746 trainable parameters.

## Train the selected MSE model

The selected configuration uses Adam, learning rate `0.0005`, batch size 32, seed 42, and early-stopping patience 5:

```bash
python3 CNN/train.py \
  --data-root shared_data \
  --loss mse \
  --learning-rate 0.0005 \
  --batch-size 32 \
  --epochs 20 \
  --patience 5 \
  --seed 42 \
  --output-dir CNN/outputs/cnn_mse
```

Training writes `best_model.pt`, `last_model.pt`, `history.json`, `config.json`, and `loss_curve.png`. Use `best_model.pt` for evaluation.

## Reproduce hyperparameter tuning

The search compared learning rates `0.0001`, `0.0002`, and `0.0005` with batch sizes 32 and 64. All other settings were fixed.

```bash
python3 CNN/tune.py \
  --data-root shared_data \
  --output-dir CNN/outputs/tuning_mse \
  --learning-rates 0.0001 0.0002 0.0005 \
  --batch-sizes 32 64 \
  --epochs 20 \
  --patience 5 \
  --seed 42
```

Runs are ranked by minimum validation MSE. The selected configuration was learning rate `0.0005`, batch size 32, with validation MSE `0.00822577` at epoch 10.

## L1 ablation

The L1 experiment uses the selected settings and changes only the reconstruction loss:

```bash
python3 CNN/train.py \
  --data-root shared_data \
  --loss l1 \
  --learning-rate 0.0005 \
  --batch-size 32 \
  --epochs 20 \
  --patience 5 \
  --seed 42 \
  --output-dir CNN/outputs/cnn_l1
```

## Visualize a checkpoint

```bash
python3 CNN/visualize.py \
  --data-root shared_data \
  --split validation \
  --checkpoint CNN/outputs/cnn_mse/best_model.pt \
  --output CNN/outputs/cnn_mse/validation_examples.png
```

## Test evaluation

```bash
python3 CNN/evaluate.py \
  --data-root shared_data \
  --checkpoint CNN/outputs/cnn_mse/best_model.pt \
  --batch-size 32 \
  --output-dir CNN/outputs/cnn_mse/test
```

## File overview

| File | Purpose |
|---|---|
| `dataset.py` | HDF5 loader and schema check |
| `model.py` | Compact U-Net definition |
| `train.py` | Training, validation, checkpointing, and early stopping |
| `tune.py` | Learning-rate and batch-size grid search |
| `visualize.py` | Train or validation comparison grids |
| `evaluate.py` | Test metrics and qualitative grids |
| `utils.py` | Device, reproducibility, Lab-to-RGB, and plotting helpers |
| `download_data.py` | Dataset download and file discovery |

CUDA, Apple MPS, and CPU are supported. Dataset files, checkpoints, and generated outputs are excluded from Git.
