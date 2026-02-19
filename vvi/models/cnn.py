from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CNNGRU(nn.Module):
    def __init__(self, cnn_feature_dim: int = 128, hidden_dim: int = 64, num_classes: int = 4) -> None:
        super().__init__()
        self.backbone = SimpleCNN(num_classes=cnn_feature_dim)
        self.gru = nn.GRU(cnn_feature_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = seq.shape
        flat = seq.view(b * t, c, h, w)
        emb = self.backbone(flat)
        emb = emb.view(b, t, -1)
        out, _ = self.gru(emb)
        return self.head(out[:, -1, :])
