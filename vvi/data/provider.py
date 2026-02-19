from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "date",
    "underlying",
    "strike",
    "days_to_expiry",
    "option_type",
    "option_price",
    "spot",
    "risk_free_rate",
}


@dataclass
class OptionsDataLoader:
    path: Path

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Options dataset not found: {self.path}")
        df = pd.read_csv(self.path, parse_dates=["date"])
        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(f"Dataset missing columns: {sorted(missing)}")
        return self._normalize(df)

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        cleaned["option_type"] = cleaned["option_type"].str.lower().str[0]
        cleaned = cleaned[cleaned["option_type"].isin(["c", "p"])].copy()
        numeric_cols = [
            "strike",
            "days_to_expiry",
            "option_price",
            "spot",
            "risk_free_rate",
        ]
        for col in numeric_cols:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        cleaned = cleaned.dropna(subset=["date", *numeric_cols]).sort_values("date")
        cleaned["days_to_expiry"] = cleaned["days_to_expiry"].clip(lower=1)
        cleaned["maturity_years"] = cleaned["days_to_expiry"] / 365.0
        cleaned["moneyness"] = cleaned["strike"] / cleaned["spot"]
        return cleaned.reset_index(drop=True)


def generate_synthetic_options_data(
    start_date: str = "2015-01-02",
    end_date: str = "2025-01-02",
    seed: int = 7,
    underlying: str = "SPX",
    sample_step: int = 5,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, end=end_date)[::sample_step]
    rows: list[dict] = []
    spot = 2000.0

    strike_grid = np.arange(0.7, 1.31, 0.05)
    maturity_days = [7, 14, 21, 30, 45, 60, 90, 120, 180]

    for date in dates:
        drift = rng.normal(0.0002, 0.01)
        spot *= float(np.exp(drift))
        base_vol = 0.12 + 0.22 * min(1.0, abs(drift) * 14)

        for m in maturity_days:
            t = m / 365.0
            term_bump = 0.02 * np.sqrt(t)
            for ratio in strike_grid:
                strike = spot * ratio
                smile = 0.18 * (ratio - 1.0) ** 2
                vol = max(0.05, base_vol + term_bump + smile + rng.normal(0.0, 0.01))
                option_type = "c" if ratio >= 1 else "p"
                intrinsic = max(0.0, spot - strike) if option_type == "c" else max(0.0, strike - spot)
                time_val = spot * vol * np.sqrt(t) * 0.35
                price = intrinsic + time_val
                rows.append(
                    {
                        "date": date,
                        "underlying": underlying,
                        "strike": strike,
                        "days_to_expiry": m,
                        "option_type": option_type,
                        "option_price": max(0.01, price),
                        "spot": spot,
                        "risk_free_rate": 0.03,
                        "implied_vol": vol,
                    }
                )

    return pd.DataFrame(rows)


def ensure_dataset(path: Path) -> pd.DataFrame:
    if path.exists():
        return OptionsDataLoader(path).load()
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_options_data()
    df.to_csv(path, index=False)
    return OptionsDataLoader(path).load()
