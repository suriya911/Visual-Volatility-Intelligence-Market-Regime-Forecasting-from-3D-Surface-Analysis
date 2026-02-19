# Visual Volatility Intelligence

End-to-end system for market regime forecasting using implied volatility surfaces rendered as images.

## What this includes

- Options data ingestion (CSV schema + synthetic fallback)
- Black-Scholes implied volatility inversion
- Daily IV surface construction and PNG rendering
- CNN training with time-series cross-validation
- Baseline comparator (logistic regression on IV features)
- Signal generation and backtest metrics (Sharpe, false-signal rate)
- FastAPI for running pipeline and serving artifacts
- React dashboard for metrics, signals, and surface visualization
- Docker and docker-compose setup

## Project structure

- `vvi/data`: ingestion and synthetic data generator
- `vvi/iv`: Black-Scholes pricing and implied volatility solver
- `vvi/surfaces`: interpolation and image rendering
- `vvi/models`: dataset, CNN model, training, baseline comparison
- `vvi/signals`: trading signal translation
- `vvi/backtest`: strategy evaluation
- `vvi/api`: REST API
- `frontend`: React dashboard
- `notebooks`: reference notebook

## Input data schema

`data/options_daily.csv` (if present) must contain:

- `date`
- `underlying`
- `strike`
- `days_to_expiry`
- `option_type` (`c` or `p`)
- `option_price`
- `spot`
- `risk_free_rate`

If this file does not exist, synthetic SPX-like data is auto-generated so you can run immediately.

## Backend quickstart

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -e .
python scripts/run_pipeline.py
python scripts/run_api.py
```

API runs on `http://localhost:8000`.

Useful endpoints:

- `POST /pipeline/run`
- `GET /metrics`
- `GET /predictions`
- `GET /surfaces`
- `GET /surfaces/{YYYY-MM-DD}`

## Frontend quickstart

```bash
cd frontend
npm install
npm run dev
```

Dashboard runs on `http://localhost:5173`.

## Docker

```bash
docker compose up --build
```

## Notes

- This implementation is production-structured and runnable.
- Replace synthetic fallback with a real historical options provider to run on full 10+ year market data.
- Optional `CNN + GRU/LSTM` hybrid class is provided in `vvi/models/cnn.py` for temporal extension.
