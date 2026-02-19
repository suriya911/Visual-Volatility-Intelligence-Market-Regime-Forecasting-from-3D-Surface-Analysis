# Implementation Checklist

- [x] 1. Scaffold modular project structure (backend package + frontend app + docs + Docker).
- [x] 2. Data ingestion pipeline for options chain snapshots and standardized schema.
- [x] 3. Black-Scholes implied volatility solver with robust edge-case handling.
- [x] 4. Daily 3D surface builder (moneyness x maturity x IV) with interpolation.
- [x] 5. Surface renderer to dated 2D image artifacts for model input.
- [x] 6. CNN training pipeline for next-day regime/direction prediction.
- [x] 7. Optional temporal model (CNN features + GRU/LSTM).
- [x] 8. Baseline model(s) and time-series cross-validation evaluation.
- [x] 9. Trading signal generator from predictions.
- [x] 10. Backtesting engine with Sharpe and false-signal metrics.
- [x] 11. FastAPI service exposing predictions, metrics, and surface assets.
- [x] 12. React dashboard showing surfaces, regimes, signals, and performance.
- [x] 13. Docker + docker-compose for reproducible local deployment.
- [x] 14. Notebook + README documenting end-to-end usage.
