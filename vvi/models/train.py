from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from torch import nn
from torch.utils.data import DataLoader

from vvi.models.cnn import SimpleCNN
from vvi.models.dataset import SurfaceImageDataset


@dataclass
class TrainConfig:
    image_dir: Path
    image_size: int = 224
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 8
    folds: int = 4
    device: str = "cpu"


def make_targets(regime_df: pd.DataFrame) -> pd.DataFrame:
    df = regime_df.sort_values("date").reset_index(drop=True).copy()
    next_change = df["mean_iv"].shift(-1) - df["mean_iv"]
    q1, q2, q3 = np.nanquantile(next_change.dropna(), [0.25, 0.5, 0.75])

    def bucket(x: float) -> int:
        if pd.isna(x):
            return -1
        if x <= q1:
            return 0
        if x <= q2:
            return 1
        if x <= q3:
            return 2
        return 3

    df["target"] = next_change.map(bucket)
    return df[df["target"] >= 0].reset_index(drop=True)


def train_timeseries_cnn(frame: pd.DataFrame, cfg: TrainConfig, model_out: Path) -> dict[str, float]:
    tscv = TimeSeriesSplit(n_splits=cfg.folds)
    dates = frame["date"].reset_index(drop=True)
    targets = frame["target"].reset_index(drop=True)

    fold_scores: list[float] = []
    device = torch.device(cfg.device)

    for train_idx, val_idx in tscv.split(np.arange(len(frame))):
        train_frame = pd.DataFrame({"date": dates.iloc[train_idx], "target": targets.iloc[train_idx]})
        val_frame = pd.DataFrame({"date": dates.iloc[val_idx], "target": targets.iloc[val_idx]})

        train_ds = SurfaceImageDataset(train_frame, cfg.image_dir, image_size=cfg.image_size)
        val_ds = SurfaceImageDataset(val_frame, cfg.image_dir, image_size=cfg.image_size)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        model = SimpleCNN(num_classes=4).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(cfg.epochs):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                opt.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(device))
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(pred.tolist())
                y_true.extend(y.numpy().tolist())

        if y_true:
            fold_scores.append(float(accuracy_score(y_true, y_pred)))

    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)

    return {
        "cnn_accuracy": float(np.mean(fold_scores) if fold_scores else 0.0),
        "cnn_accuracy_std": float(np.std(fold_scores) if fold_scores else 0.0),
    }


def infer(model_path: Path, frame: pd.DataFrame, image_dir: Path, image_size: int = 224) -> pd.DataFrame:
    model = SimpleCNN(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    ds = SurfaceImageDataset(frame[["date", "target"]], image_dir, image_size=image_size)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    preds: list[int] = []
    probs: list[float] = []

    with torch.no_grad():
        for x, _ in loader:
            logits = model(x)
            p = torch.softmax(logits, dim=1)
            pred = torch.argmax(p, dim=1)
            preds.extend(pred.numpy().tolist())
            probs.extend(torch.max(p, dim=1).values.numpy().tolist())

    out = frame.copy().reset_index(drop=True)
    out["pred_regime"] = preds
    out["pred_confidence"] = probs
    return out
