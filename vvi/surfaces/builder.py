from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from vvi.iv import implied_volatility


@dataclass
class SurfaceConfig:
    moneyness_bins: int = 48
    maturity_bins: int = 48


class SurfaceBuilder:
    def __init__(self, config: SurfaceConfig | None = None) -> None:
        self.config = config or SurfaceConfig()

    def add_implied_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        enriched = df.copy()
        if "implied_vol" in enriched.columns:
            enriched["implied_vol"] = pd.to_numeric(enriched["implied_vol"], errors="coerce")
            missing_mask = enriched["implied_vol"].isna()
        else:
            missing_mask = pd.Series(True, index=enriched.index)

        if missing_mask.any():
            enriched.loc[missing_mask, "implied_vol"] = enriched.loc[missing_mask].apply(
                lambda r: implied_volatility(
                    option_price=float(r.option_price),
                    spot=float(r.spot),
                    strike=float(r.strike),
                    t=float(r.maturity_years),
                    rate=float(r.risk_free_rate),
                    option_type=str(r.option_type),
                ),
                axis=1,
            )
        return enriched.dropna(subset=["implied_vol"]).reset_index(drop=True)

    def build_daily_surface(self, day_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = day_df["moneyness"].to_numpy()
        y = day_df["maturity_years"].to_numpy()
        z = day_df["implied_vol"].to_numpy()

        x_grid = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), self.config.moneyness_bins)
        y_grid = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), self.config.maturity_bins)
        xx, yy = np.meshgrid(x_grid, y_grid)

        surface = griddata((x, y), z, (xx, yy), method="linear")
        if np.isnan(surface).any():
            fill = griddata((x, y), z, (xx, yy), method="nearest")
            surface = np.where(np.isnan(surface), fill, surface)

        return xx, yy, surface

    def build_all_surfaces(self, df: pd.DataFrame) -> dict[pd.Timestamp, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        surfaces: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for date, day_df in df.groupby("date"):
            if len(day_df) < 10:
                continue
            surfaces[pd.Timestamp(date)] = self.build_daily_surface(day_df)
        return surfaces

    @staticmethod
    def export_surface_matrix(path: Path, date: pd.Timestamp, surface: np.ndarray) -> None:
        path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(surface).to_csv(path / f"{date.date()}.csv", index=False)
