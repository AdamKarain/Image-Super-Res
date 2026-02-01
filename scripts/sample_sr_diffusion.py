from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import torch
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.celebahq_dataset import CelebAHQDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample diffusion SR model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--input_image", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./reports/figures")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--scheduler", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def load_checkpoint(path: str | Path) -> tuple[UNet2DModel, dict]:
    ckpt = torch.load(path, map_location="cpu")
    model = UNet2DModel(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def make_lr_from_image(image: Image.Image, image_size: int, downscale: int) -> torch.Tensor:
    hr_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    lr_size = image_size // downscale
    down = transforms.Resize(
        (lr_size, lr_size),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )
    up = transforms.Resize(
        (image_size, image_size),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )
    hr = hr_transform(image)
    lr = up(down(image))
    lr = transforms.ToTensor()(lr)
    lr = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(lr)
    return hr, lr


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
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

    if args.input_image:
        image = Image.open(args.input_image).convert("RGB")
        hr, lr = make_lr_from_image(image, image_size, downscale)
        image_name = Path(args.input_image).stem
    else:
        if not args.dataset_root:
            raise ValueError("dataset_root is required when input_image is not set.")
        dataset = CelebAHQDataset(
            root_dir=args.dataset_root,
            image_size=image_size,
            downscale=downscale,
            split="val",
        )
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]
        hr = sample["hr"]
        lr = sample["lr"]
        image_name = Path(sample["path"]).stem

    hr = hr.unsqueeze(0).to(device)
    lr = lr.unsqueeze(0).to(device)
    sample = torch.randn_like(hr)

    for t in scheduler.timesteps:
        model_input = torch.cat([sample, lr], dim=1)
        with torch.no_grad():
            noise_pred = model(model_input, t).sample
        sample = scheduler.step(noise_pred, t, sample).prev_sample

    def denorm(x: torch.Tensor) -> torch.Tensor:
        return (x.clamp(-1, 1) + 1) / 2

    grid = torch.cat([denorm(lr), denorm(sample), denorm(hr)], dim=0)
    save_path = output_dir / f"{image_name}_sr.png"
    save_image(grid, save_path, nrow=3)


if __name__ == "__main__":
    main()
