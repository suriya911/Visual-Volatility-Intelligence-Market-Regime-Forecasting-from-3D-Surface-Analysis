from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def evaluate_baseline(frame: pd.DataFrame) -> dict[str, float]:
    feature_df = frame.copy()
    feature_df["prev_iv"] = feature_df["mean_iv"].shift(1)
    feature_df["iv_change"] = feature_df["mean_iv"].diff()
    feature_df = feature_df.dropna().reset_index(drop=True)

    split = int(len(feature_df) * 0.8)
    train, test = feature_df.iloc[:split], feature_df.iloc[split:]
    if train.empty or test.empty:
        return {"baseline_accuracy": 0.0}

    x_train = train[["prev_iv", "iv_change"]].to_numpy()
    y_train = train["target"].to_numpy()
    x_test = test[["prev_iv", "iv_change"]].to_numpy()
    y_test = test["target"].to_numpy()

    clf = LogisticRegression(max_iter=400)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return {"baseline_accuracy": float(accuracy_score(y_test, pred))}
