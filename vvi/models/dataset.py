from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SurfaceImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_dir: Path, image_size: int = 224) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[idx]
        image_path = self.image_dir / f"{pd.Timestamp(row['date']).date()}.png"
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image)
        y = torch.tensor(int(row["target"]), dtype=torch.long)
        return x, y
