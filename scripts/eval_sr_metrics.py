from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel
from torch.utils.data import DataLoader
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.celebahq_dataset import CelebAHQDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SR diffusion metrics.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--scheduler", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--num_samples", type=int, default=100)
    return parser.parse_args()


def load_checkpoint(path: str | Path) -> tuple[UNet2DModel, dict]:
    ckpt = torch.load(path, map_location="cpu")
    model = UNet2DModel(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) / 2


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt = load_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    image_size = ckpt["image_size"]
    downscale = ckpt["downscale"]
    num_timesteps = ckpt["num_timesteps"]
    if args.scheduler == "ddim":
        scheduler = DDIMScheduler(num_train_timesteps=num_timesteps)
    else:
        scheduler = DDPMScheduler(num_train_timesteps=num_timesteps)
    scheduler.set_timesteps(args.num_steps, device=device)

    dataset = CelebAHQDataset(
        root_dir=args.dataset_root,
        image_size=image_size,
        downscale=downscale,
        split=args.split,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    total_psnr = 0.0
    total_ssim = 0.0
    seen = 0

    for batch in dataloader:
        hr = batch["hr"].to(device)
        lr = batch["lr"].to(device)
        sample = torch.randn_like(hr)

        for t in scheduler.timesteps:
            model_input = torch.cat([sample, lr], dim=1)
            with torch.no_grad():
                noise_pred = model(model_input, t).sample
            sample = scheduler.step(noise_pred, t, sample).prev_sample

        sr = denorm(sample)
        hr = denorm(hr)

        batch_psnr = peak_signal_noise_ratio(sr, hr, data_range=1.0)
        batch_ssim = structural_similarity_index_measure(sr, hr, data_range=1.0)
        total_psnr += batch_psnr.item() * hr.size(0)
        total_ssim += batch_ssim.item() * hr.size(0)
        seen += hr.size(0)
        if seen >= args.num_samples:
            break

    avg_psnr = total_psnr / max(seen, 1)
    avg_ssim = total_ssim / max(seen, 1)
    print(f"Samples: {seen}")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")


if __name__ == "__main__":
    main()
