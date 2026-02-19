from __future__ import annotations

import pandas as pd


def generate_signals(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()

    def map_signal(regime: int, confidence: float) -> int:
        if confidence < 0.35:
            return 0
        if regime in (2, 3):
            return 1
        if regime == 0:
            return -1
        return 0

    out["signal"] = [map_signal(r, c) for r, c in zip(out["pred_regime"], out["pred_confidence"])]
    return out
