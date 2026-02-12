# Diffusion Based Image Super Resolution

[Open in
Colab](https://colab.research.google.com/github/AdamKarain/Image-Super-Res/blob/main/notebooks/colab_train.ipynb)

This project implements image super resolution using a conditional
diffusion model.\
The goal is to reconstruct a sharp high resolution face image from a
blurred low resolution version.

We trained the model on the CelebA-HQ 256×256 dataset and evaluated
reconstruction quality using PSNR and SSIM.

## How the model works (simple explanation)

Instead of directly predicting a high resolution image, the model learns
to remove noise step-by-step.

During training: 1. We take a real high resolution image 2. Add random
noise to it 3. Give the model: - the noisy image - the low resolution
version of the same image 4. The model learns to predict the noise that
was added

At inference time we start from noise and gradually reconstruct a clean
high resolution image conditioned on the low resolution input.

## Dataset

We used:
https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256

Folder structure: data/ celeba_hq_256/ 00000.jpg 00001.jpg

We split the dataset automatically inside the dataset loader: 95%
training 5% validation

## Training (Colab)

Training was done on Google Colab GPU and checkpoints were saved to
Google Drive so training could be resumed.

Example command: python scripts/train_sr_diffusion.py --dataset_root
./data/celeba_hq_256 --output_dir checkpoints --image_size 256
--downscale 4 --batch_size 1 --epochs 10

## Evaluation

We measured reconstruction quality using PSNR and SSIM on validation
images.

Example result: PSNR ≈ 18.5 SSIM ≈ 0.63
