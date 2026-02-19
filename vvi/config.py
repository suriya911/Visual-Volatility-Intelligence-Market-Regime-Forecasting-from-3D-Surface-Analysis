from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    raw_data_path: Path = Path("data/options_fast.csv")
    processed_path: Path = Path("data/processed_options.csv")
    surfaces_dir: Path = Path("artifacts/surfaces")
    model_dir: Path = Path("artifacts/models")
    metrics_path: Path = Path("artifacts/metrics.json")
    prediction_path: Path = Path("artifacts/predictions.csv")
    image_size: int = 224
    epochs: int = 4
    batch_size: int = 32
    lr: float = 1e-3
    sequence_length: int = 5
    max_surface_days: int = 240
