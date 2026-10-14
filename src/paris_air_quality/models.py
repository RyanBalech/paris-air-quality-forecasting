"""Direct 21-day forecasts with seasonal and weather-aware models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .data import TARGETS
from .features import HORIZON_HOURS, build_features


def seasonal_median(history: pd.DataFrame, future: pd.DatetimeIndex) -> pd.DataFrame:
    """Same-hour-of-week median from the last eight weeks, with robust fallback."""
    recent = history.loc[history.index > history.index[-1] - pd.Timedelta(weeks=8)]
    output = pd.DataFrame(index=future, columns=TARGETS, dtype=float)
    key = recent.index.dayofweek * 24 + recent.index.hour
    future_key = future.dayofweek * 24 + future.hour
    for target in TARGETS:
        fallback = float(recent[target].median())
        if not np.isfinite(fallback):
            fallback = float(history[target].median())
        profile = recent[target].groupby(key).median()
        output[target] = profile.reindex(future_key).fillna(fallback).to_numpy()
    return output.clip(lower=0)


def direct_lightgbm(history: pd.DataFrame, future: pd.DatetimeIndex,
                    weather: pd.DataFrame | None = None, *, train_years: int = 3,
                    n_estimators: int = 500) -> pd.DataFrame:
    """Train one median-regression model per pollutant without future pollutant features."""
    if len(future) != HORIZON_HOURS or future[0] != history.index[-1] + pd.Timedelta(hours=1):
        raise ValueError("Expected a contiguous 504-hour horizon after the training cutoff")
    start = history.index[-1] - pd.DateOffset(years=train_years)
    labels = history.loc[history.index >= start]
    x_train = build_features(labels.index, history, weather)
    x_future = build_features(future, history, weather)
    days_old = (history.index[-1] - labels.index).total_seconds() / 86400
    weights = np.asarray(np.exp(-days_old / 730), dtype=float)
    output = pd.DataFrame(index=future, columns=TARGETS, dtype=float)
    for target in TARGETS:
        valid = labels[target].notna().to_numpy()
        model = LGBMRegressor(
            objective="regression_l1", n_estimators=n_estimators, learning_rate=0.035,
            num_leaves=31, min_child_samples=80, colsample_bytree=0.9,
            subsample=0.8, subsample_freq=1, reg_lambda=2.0,
            random_state=42, n_jobs=4, verbosity=-1,
        )
        model.fit(x_train.loc[valid], labels.loc[valid, target], sample_weight=weights[valid])
        output[target] = model.predict(x_future)
    return output.clip(lower=0)
