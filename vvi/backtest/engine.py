from __future__ import annotations

import math

import numpy as np
import pandas as pd


def run_backtest(signal_df: pd.DataFrame) -> dict[str, float]:
    df = signal_df.sort_values("date").copy()
    df["iv_return"] = df["mean_iv"].shift(-1) - df["mean_iv"]
    df = df.dropna(subset=["iv_return"]).reset_index(drop=True)

    df["strategy_return"] = df["signal"] * df["iv_return"]
    mean_ret = float(df["strategy_return"].mean())
    std_ret = float(df["strategy_return"].std(ddof=1)) if len(df) > 1 else 0.0
    sharpe = (mean_ret / std_ret) * math.sqrt(252) if std_ret > 0 else 0.0

    false_signal = float((np.sign(df["signal"]) != np.sign(df["iv_return"])).mean())

    long_only = df["iv_return"].copy()
    lo_std = float(long_only.std(ddof=1)) if len(long_only) > 1 else 0.0
    lo_sharpe = (float(long_only.mean()) / lo_std) * math.sqrt(252) if lo_std > 0 else 0.0

    return {
        "strategy_sharpe": sharpe,
        "baseline_sharpe": lo_sharpe,
        "sharpe_improvement_pct": ((sharpe - lo_sharpe) / abs(lo_sharpe) * 100.0) if lo_sharpe != 0 else 0.0,
        "false_signal_rate": false_signal,
    }
