# Image Super-Resolution (Diffusion-Based)

This project trains a conditional diffusion model for image super-resolution
using the CelebA-HQ resized dataset from Kaggle.

## Dataset

Download the dataset from Kaggle:
`https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256`

Recommended layout:

```
Image-Super-Res/
  data/
    celebahq-resized-256x256/
      00000.png
      00001.png
      ...
```

If you have the Kaggle CLI configured:

```
kaggle datasets download -d badasstechie/celebahq-resized-256x256 -p ./data
unzip ./data/celebahq-resized-256x256.zip -d ./data/celebahq-resized-256x256
```

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```
python scripts/train_sr_diffusion.py \
  --dataset_root ./data/celebahq-resized-256x256 \
  --output_dir ./reports/checkpoints \
  --image_size 256 \
  --downscale 4 \
  --batch_size 8 \
  --epochs 10
```

## Sampling / Inference

```
python scripts/sample_sr_diffusion.py \
  --checkpoint ./reports/checkpoints/sr_diffusion_latest.pt \
  --dataset_root ./data/celebahq-resized-256x256 \
  --output_dir ./reports/figures \
  --num_steps 50 \
  --scheduler ddim
```

To use a specific image:

```
python scripts/sample_sr_diffusion.py \
  --checkpoint ./reports/checkpoints/sr_diffusion_latest.pt \
  --input_image ./data/celebahq-resized-256x256/00010.png \
  --output_dir ./reports/figures \
  --num_steps 50 \
  --scheduler ddim

## Evaluation (PSNR/SSIM)

```
python scripts/eval_sr_metrics.py \
  --checkpoint ./reports/checkpoints/sr_diffusion_latest.pt \
  --dataset_root ./data/celebahq-resized-256x256 \
  --num_steps 50 \
  --scheduler ddim \
  --num_samples 100
```
```

## Notes

- The model is conditioned on the low-resolution image by concatenating it with
  the noisy high-resolution image (6 input channels total).
- Low-resolution inputs are created by downscaling and then upscaling with
  bicubic interpolation to match the high-resolution size.
