from __future__ import annotations

from pathlib import Path
import random
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _list_images(root_dir: str | Path) -> List[Path]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    images = [
        path
        for path in root.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"No images found under: {root}")
    return sorted(images)


def _split_paths(
    paths: List[Path],
    train_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    indices = list(range(len(paths)))
    rng.shuffle(indices)
    split_idx = int(len(indices) * train_ratio)
    train_paths = [paths[i] for i in indices[:split_idx]]
    val_paths = [paths[i] for i in indices[split_idx:]]
    return train_paths, val_paths


class CelebAHQDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 256,
        downscale: int = 4,
        split: str = "train",
        train_ratio: float = 0.95,
        seed: int = 42,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if image_size % downscale != 0:
            raise ValueError("image_size must be divisible by downscale")
        self.image_size = image_size
        self.downscale = downscale
        all_paths = _list_images(root_dir)
        train_paths, val_paths = _split_paths(all_paths, train_ratio, seed)
        self.paths = train_paths if split == "train" else val_paths
        self.hr_transform = transforms.Compose(
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
        self.lr_down = transforms.Resize(
            (lr_size, lr_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )
        self.lr_up = transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        hr = self.hr_transform(image)
        lr = self.lr_up(self.lr_down(image))
        lr = transforms.ToTensor()(lr)
        lr = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(lr)
        return {"lr": lr, "hr": hr, "path": str(path)}


def make_lr_from_hr(
    hr_tensor: torch.Tensor,
    downscale: int,
) -> torch.Tensor:
    if hr_tensor.dim() != 3:
        raise ValueError("hr_tensor must be CHW")
    _, h, w = hr_tensor.shape
    lr_h = h // downscale
    lr_w = w // downscale
    down = transforms.Resize(
        (lr_h, lr_w),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )
    up = transforms.Resize(
        (h, w),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )
    lr = up(down(hr_tensor))
    return lr
