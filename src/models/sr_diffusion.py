from __future__ import annotations

from diffusers import UNet2DModel


def create_sr_unet(image_size: int) -> UNet2DModel:
    return UNet2DModel(
        sample_size=image_size,
        in_channels=6,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
