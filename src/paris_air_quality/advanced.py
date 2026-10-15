"""Alternative models for weather, human activity and horizon-specific histories."""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .data import TARGETS
from .features import build_features


def features(index: pd.DatetimeIndex, history: pd.DataFrame, weather: pd.DataFrame,
             lag_floor: int = 504, use_history: bool = True) -> pd.DataFrame:
    if lag_floor not in (168, 336, 504):
        raise ValueError("Expected a one, two or three-week lag floor")
    x = build_features(index, history, weather, enhanced=True)
    x = x.loc[:, ~x.columns.str.startswith(TARGETS)]
    extra = {}
    if use_history:
        for target in TARGETS:
            lag_values = []
            for offset in range(6):
                lag = lag_floor + offset * 168
                values = history[target].reindex(index - pd.Timedelta(hours=lag)).to_numpy()
                extra[f"{target}_lag_{lag}"] = values
                lag_values.append(values)
            extra[f"{target}_weekly_median"] = pd.DataFrame(lag_values).median(axis=0).to_numpy()
            for window in (24, 168, 672):
                extra[f"{target}_mean_{window}"] = history[target].rolling(
                    window, min_periods=window // 2).mean().reindex(
                        index - pd.Timedelta(hours=lag_floor)).to_numpy()
    calendar = holidays.France(years=range(index.year.min() - 1, index.year.max() + 2))
    extra["holiday"] = np.array([t.date() in calendar for t in index], dtype=float)
    extra["holiday_previous"] = np.array([
        (t - pd.Timedelta(days=1)).date() in calendar for t in index], dtype=float)
    extra["holiday_next"] = np.array([
        (t + pd.Timedelta(days=1)).date() in calendar for t in index], dtype=float)
    extra["workday"] = ((index.dayofweek < 5) & (extra["holiday"] == 0)).astype(float)
    w = weather.reindex(index)
    # Boundary-layer height has a six-month API gap. Exclude it consistently rather
    # than introduce an unreliable imputation into the comparison.
    x = x.loc[:, ~x.columns.str.contains("boundary_layer_height")]
    direction = np.deg2rad(w["wind_direction_100m"])
    extra["wind_east_100m"] = w["wind_speed_100m"] * np.sin(direction)
    extra["wind_north_100m"] = w["wind_speed_100m"] * np.cos(direction)
    extra["wind_shear"] = w["wind_speed_100m"] - w["wind_speed_10m"]
    extra["dewpoint_depression"] = w["temperature_2m"] - w["dew_point_2m"]
    for column in ("wind_speed_100m", "dew_point_2m", "direct_radiation"):
        for window in (6, 24):
            extra[f"{column}_mean{window}"] = weather[column].rolling(
                window, min_periods=window // 2).mean().reindex(index)
    return pd.concat([x, pd.DataFrame(extra, index=index)], axis=1).astype("float32")


def advanced_forecast(history: pd.DataFrame, future: pd.DatetimeIndex, weather: pd.DataFrame,
                      *, family: str = "lightgbm", use_history: bool = True,
                      weekly: bool = False, iterations: int = 800) -> pd.DataFrame:
    if not future.equals(pd.date_range(history.index[-1] + pd.Timedelta(hours=1),
                                       periods=504, freq="h")):
        raise ValueError("Expected a contiguous 504-hour forecast horizon")
    labels = history.loc[history.index >= history.index[-1] - pd.DateOffset(years=3)]
    days_old = (history.index[-1] - labels.index).total_seconds() / 86400
    weights = np.asarray(np.exp(-days_old / 730), dtype=float)
    result = pd.DataFrame(index=future, columns=TARGETS, dtype=float)
    for week in (range(3) if weekly else [2]):
        lag_floor = (week + 1) * 168
        prediction_index = future[week * 168:(week + 1) * 168] if weekly else future
        x_train = features(labels.index, history, weather, lag_floor, use_history)
        x_future = features(prediction_index, history, weather, lag_floor, use_history)
        for target in TARGETS:
            valid = labels[target].notna()
            if family == "catboost":
                from catboost import CatBoostRegressor
                model = CatBoostRegressor(
                    iterations=iterations, depth=6, learning_rate=.05, loss_function="MAE",
                    l2_leaf_reg=5, random_seed=42, thread_count=4,
                    allow_writing_files=False, verbose=False,
                )
            elif family == "lightgbm":
                model = LGBMRegressor(
                    objective="regression_l1", n_estimators=iterations, learning_rate=.035,
                    num_leaves=15, min_child_samples=120, colsample_bytree=.9,
                    subsample=.8, subsample_freq=1, reg_lambda=2,
                    random_state=42, n_jobs=4, verbosity=-1,
                )
            else:
                raise ValueError(f"Unknown family: {family}")
            model.fit(x_train.loc[valid], labels.loc[valid, target],
                      sample_weight=weights[valid])
            result.loc[prediction_index, target] = model.predict(x_future)
    return result.clip(lower=0)
