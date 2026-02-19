from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from vvi.config import PipelineConfig
from vvi.pipeline import run_pipeline

cfg = PipelineConfig()
app = FastAPI(title="Visual Volatility Intelligence API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/pipeline/run")
def pipeline_run() -> dict:
    return run_pipeline(cfg)


@app.get("/metrics")
def get_metrics() -> dict:
    if not cfg.metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics not found. Run pipeline first.")
    return json.loads(cfg.metrics_path.read_text())


@app.get("/predictions")
def get_predictions(limit: int = 100) -> list[dict]:
    if not cfg.prediction_path.exists():
        raise HTTPException(status_code=404, detail="Predictions not found. Run pipeline first.")
    df = pd.read_csv(cfg.prediction_path)
    return df.tail(limit).to_dict(orient="records")


@app.get("/surfaces/{date_str}")
def get_surface(date_str: str):
    img = cfg.surfaces_dir / f"{date_str}.png"
    if not img.exists():
        raise HTTPException(status_code=404, detail="Surface image not found")
    return FileResponse(img)


@app.get("/surfaces")
def list_surfaces(limit: int = 30) -> list[str]:
    if not cfg.surfaces_dir.exists():
        return []
    files = sorted([p.name for p in cfg.surfaces_dir.glob("*.png")])
    return files[-limit:]


def run() -> None:
    import uvicorn

    uvicorn.run("vvi.api.main:app", host="0.0.0.0", port=8000, reload=False)
