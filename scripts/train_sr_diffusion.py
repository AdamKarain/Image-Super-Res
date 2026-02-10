from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import DDPMScheduler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.celebahq_dataset import CelebAHQDataset
from src.models.sr_diffusion import create_sr_unet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train diffusion SR model.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./reports/checkpoints")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CelebAHQDataset(
        root_dir=args.dataset_root,
        image_size=args.image_size,
        downscale=args.downscale,
        split="train",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model = create_sr_unet(args.image_size).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=args.num_timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            hr = batch["hr"].to(device)
            lr = batch["lr"].to(device)
            noise = torch.randn_like(hr)
            timesteps = torch.randint(
                0,
                scheduler.num_train_timesteps,
                (hr.shape[0],),
                device=device,
            ).long()
            noisy_hr = scheduler.add_noise(hr, noise, timesteps)
            model_input = torch.cat([noisy_hr, lr], dim=1)
            noise_pred = model(model_input, timesteps).sample
            loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        if epoch % args.save_every == 0:
            ckpt_path = output_dir / f"sr_diffusion_epoch_{epoch}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": dict(model.config),
                    "image_size": args.image_size,
                    "downscale": args.downscale,
                    "num_timesteps": args.num_timesteps,
                },
                ckpt_path,
            )

    latest_path = output_dir / "sr_diffusion_latest.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": dict(model.config),
            "image_size": args.image_size,
            "downscale": args.downscale,
            "num_timesteps": args.num_timesteps,
        },
        latest_path,
    )


if __name__ == "__main__":
    main()
