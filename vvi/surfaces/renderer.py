from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SurfaceRenderer:
    def __init__(self, image_size: int = 224, cmap: str = "viridis") -> None:
        self.image_size = image_size
        self.cmap = cmap

    def render(self, date: pd.Timestamp, xx: np.ndarray, yy: np.ndarray, surface: np.ndarray, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{date.date()}.png"

        fig = plt.figure(figsize=(4, 4), dpi=self.image_size // 4)
        ax = fig.add_subplot(111)
        im = ax.imshow(surface, origin="lower", aspect="auto", cmap=self.cmap)
        ax.set_title(str(date.date()))
        ax.set_xlabel("Moneyness")
        ax.set_ylabel("Maturity")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
