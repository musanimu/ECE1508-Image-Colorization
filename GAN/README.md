# Conditional GAN colorization

This implementation trains a conditional GAN to predict normalized CIELAB chrominance from luminance:

```text
L [B, 1, 32, 32] -> G -> ab [B, 2, 32, 32]
D(L, ab) -> real / fake logits
```

The generator is a U-Net (first conv stride 1, bottleneck at 2×2, 1×1 tanh head). The discriminator is a PatchGAN / image discriminator selected by receptive-field size (`4`, `16`, or `32`; currently hardcoded to `32` in `train.py`). Predicted `ab` is combined with input `L` only when reconstructing RGB for metrics and plots.

The canonical generator weights file is `GAN/CGAN.pt`. Training, sampling, and the project-root benchmark all use it by default.

## Data

The shared preprocessing pipeline provides fixed CIFAR-10 splits of 45,000 training, 5,000 validation, and 10,000 test images. Each HDF5 file contains:


| Key              | Shape            | Stored dtype |
| ---------------- | ---------------- | ------------ |
| `L`              | `[N, 1, 32, 32]` | `float16`    |
| `ab`             | `[N, 2, 32, 32]` | `float16`    |
| `labels`         | `[N]`            | integer      |
| `source_indices` | `[N]`            | integer      |


The stored normalization is:

```text
L_normalized = L / 50 - 1
ab_normalized = ab / 128
```

`train.py` loads `L` and `ab` as `float32` without further preprocessing. Paths are resolved by the repository-root `download_data.py` helper (prefers `shared_data/`, then legacy extract folders). The GAN experiments do not rerun `preprocess_image_colorization_data.py`.

## Setup

Run all commands from the repository root (`final_project/`):

```bash
python3 -m pip install -r GAN/requirements.txt
python3 download_data.py --output-dir shared_data
```

Check data and the model interface:

```bash
find shared_data -name "*_lab.h5"
python3 -c "from GAN.model import CGAN; import torch; m=CGAN(patch_size=32); y=m(torch.randn(4,1,32,32)); print(y.shape)"
```

Parameter counts:


| Module            | Parameters |
| ----------------- | ---------- |
| Generator (U-Net) | 16,653,570 |
| Patch4 D          | 11,841     |
| Patch16 D         | 136,641    |
| Patch32 (image) D | 2,766,529  |




## Losses

Discriminator (BCE with logits; optional one-sided label smoothing on real targets):

```text
L_D = 0.5 * (BCE(D(L, ab_real), y_real) + BCE(D(L, ab_fake.detach()), 0))
```

Generator (non-saturating GAN + L1):

```text
L_G = BCE(D(L, G(L)), 1) + λ * ||G(L) - ab||_1
```



## Train

CLI flags: `--epochs`, `--batch-size`, `--out-dir`, `--lr`, `--lambda-l1`, `--lr-decay`, `--label-smoothing`, `--resume`, `--from-scratch`.

Hardcoded in `train.py`: `PATCH_SIZE = 32`, `BETA1 = 0.0`, `D_LR_MULT = 0.1` (`lr_D = 0.1 * lr_G`). Each batch runs **one D step, then one G step**.

```bash
python3 -m GAN.train \
  --epochs 50 \
  --batch-size 64 \
  --lr 2e-4 \
  --lambda-l1 100 \
  --out-dir GAN/checkpoints_run
```

By default `--resume` points at `GAN/CGAN.pt`. If that file exists, training loads **generator weights only** (D and optimizers always start fresh) and continues from epoch 1 numbering for the new run. If the file is missing, training starts from scratch.


| Flag                    | Meaning                                                 |
| ----------------------- | ------------------------------------------------------- |
| `--lr-decay`            | Linearly decay LR to 0 over this run                    |
| `--label-smoothing 0.1` | Real targets become `0.9` for D                         |
| `--resume PATH`         | Generator checkpoint to load (default: `GAN/CGAN.pt`)   |
| `--from-scratch`        | Ignore `CGAN.pt` / `--resume` and train a new generator |




### Checkpoints

Only generator weights are saved (`G` key in the checkpoint dict):


| File                  | When                                               |
| --------------------- | -------------------------------------------------- |
| `GAN/CGAN.pt`         | Overwritten whenever validation **LPIPS** improves |
| `<out-dir>/latest.pt` | Every epoch                                        |


There is no separate `best.pt` / `best_l1.pt`. Each epoch also appends `metrics.csv` and writes `samples/epoch_XXXX.png`.

Fine-tune (auto-loads `CGAN.pt` if present):

```bash
python3 -m GAN.train \
  --epochs 50 \
  --batch-size 64 \
  --lr 2e-4 \
  --lambda-l1 60 \
  --out-dir GAN/checkpoints_finetune
```

Train from scratch even when `CGAN.pt` exists:

```bash
python3 -m GAN.train --from-scratch --epochs 50 --out-dir GAN/checkpoints_scratch
```

## Cross-model benchmark

Project-root `eval_benchmark.py` evaluates CGAN (and later CNN / CVAE) on a shared test draw: RGB MAE, MSE, PSNR, and Hasler–Süsstrunk colorfulness, plus an L | CNN | CVAE | CGAN | GT grid. CGAN weights default to `GAN/CGAN.pt`. CLI is only `--seed` and `--num-samples`:

```bash
python3 eval_benchmark.py --seed 42 --num-samples 100
```



## File overview


| File               | Purpose                                             |
| ------------------ | --------------------------------------------------- |
| `generator.py`     | U-Net generator `L → ab`                            |
| `discriminator.py` | Patch4 / Patch16 / Patch32 discriminators           |
| `model.py`         | `CGAN` wrapper with D / G losses                    |
| `train.py`         | Dataset, training loop, LPIPS selection → `CGAN.pt` |
| `CGAN.pt`          | Canonical best generator weights (by val LPIPS)     |
| `__init__.py`      | Package exports                                     |


Related repo-root scripts: `download_data.py`, `eval_benchmark.py`.

CUDA is used when available; otherwise training and eval fall back to CPU. Dataset files, checkpoints (`*.pt`), and large generated artifacts are excluded from Git.