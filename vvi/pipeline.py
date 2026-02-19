from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vvi.backtest import run_backtest
from vvi.config import PipelineConfig
from vvi.data import ensure_dataset
from vvi.models import TrainConfig, evaluate_baseline, infer, make_targets, train_timeseries_cnn
from vvi.signals import generate_signals
from vvi.surfaces import SurfaceBuilder, SurfaceRenderer


def run_pipeline(config: PipelineConfig | None = None) -> dict[str, float]:
    cfg = config or PipelineConfig()

    df = ensure_dataset(cfg.raw_data_path)
    unique_dates = sorted(pd.to_datetime(df["date"]).unique())
    if len(unique_dates) > cfg.max_surface_days:
        step = max(1, len(unique_dates) // cfg.max_surface_days)
        keep = set(unique_dates[::step])
        df = df[df["date"].isin(keep)].reset_index(drop=True)

    cfg.processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.processed_path, index=False)

    builder = SurfaceBuilder()
    iv_df = builder.add_implied_vol(df)

    regime_df = (
        iv_df.groupby("date", as_index=False)["implied_vol"]
        .mean()
        .rename(columns={"implied_vol": "mean_iv"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    surfaces = builder.build_all_surfaces(iv_df)
    renderer = SurfaceRenderer(image_size=cfg.image_size)
    for date, (xx, yy, surf) in surfaces.items():
        renderer.render(date, xx, yy, surf, cfg.surfaces_dir)

    labeled = make_targets(regime_df)
    labeled = labeled[labeled["date"].isin(pd.to_datetime(list(surfaces.keys())))]
    labeled = labeled.reset_index(drop=True)

    train_cfg = TrainConfig(
        image_dir=cfg.surfaces_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        epochs=cfg.epochs,
    )

    model_path = cfg.model_dir / "cnn.pt"
    cnn_metrics = train_timeseries_cnn(labeled[["date", "target"]], train_cfg, model_path)
    baseline_metrics = evaluate_baseline(labeled)

    pred_df = infer(model_path, labeled[["date", "target", "mean_iv"]], cfg.surfaces_dir, image_size=cfg.image_size)
    signal_df = generate_signals(pred_df)
    bt_metrics = run_backtest(signal_df)

    cfg.prediction_path.parent.mkdir(parents=True, exist_ok=True)
    signal_df.to_csv(cfg.prediction_path, index=False)

    metrics = {**cnn_metrics, **baseline_metrics, **bt_metrics}
    cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    metrics = run_pipeline()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
