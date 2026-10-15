"""Forecast features available throughout a 504-hour horizon."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TARGETS

HORIZON_HOURS = 504
TARGET_LAGS = (504, 672, 840, 1008, 1176, 1344)


def build_features(index: pd.DatetimeIndex, known_targets: pd.DataFrame,
                   weather: pd.DataFrame | None = None, *, enhanced: bool = False) -> pd.DataFrame:
    """Build features; all pollutant lags are at least the forecast horizon."""
    if any(lag < HORIZON_HOURS for lag in TARGET_LAGS):
        raise AssertionError("Target lag would leak future validation labels")
    x = pd.DataFrame(index=index)
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    doy = index.dayofyear.to_numpy()
    x["hour"] = hour
    x["day_of_week"] = dow
    x["day_of_year"] = doy
    x["month"] = index.month
    x["year"] = index.year
    x["is_weekend"] = (dow >= 5).astype(int)
    x["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    x["week_sin"] = np.sin(2 * np.pi * dow / 7)
    x["week_cos"] = np.cos(2 * np.pi * dow / 7)
    x["year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    x["year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for target in TARGETS:
        for lag in TARGET_LAGS:
            x[f"{target}_lag_{lag}"] = known_targets[target].reindex(
                index - pd.Timedelta(hours=lag)).to_numpy()
    if enhanced:
        extra = {}
        for target in TARGETS:
            series = known_targets[target]
            for window in (24, 168, 672):
                extra[f"{target}_past_mean{window}"] = series.rolling(
                    window, min_periods=window // 2).mean().reindex(
                        index - pd.Timedelta(hours=HORIZON_HOURS)).to_numpy()
            lag_columns = [f"{target}_lag_{lag}" for lag in TARGET_LAGS]
            extra[f"{target}_weekly_median"] = x[lag_columns].median(axis=1)
        x = pd.concat([x, pd.DataFrame(extra, index=index)], axis=1)
    if weather is not None:
        w = weather.reindex(index)
        for column in w.columns:
            x[f"weather_{column}"] = w[column].to_numpy()
        direction = np.deg2rad(w["wind_direction_10m"].to_numpy())
        x["wind_east"] = w["wind_speed_10m"].to_numpy() * np.sin(direction)
        x["wind_north"] = w["wind_speed_10m"].to_numpy() * np.cos(direction)
        for column in ("temperature_2m", "precipitation", "wind_speed_10m",
                       "relative_humidity_2m", "shortwave_radiation"):
            series = weather[column]
            for lag in (3, 6, 24):
                x[f"weather_{column}_lag{lag}"] = series.reindex(
                    index - pd.Timedelta(hours=lag)).to_numpy()
            x[f"weather_{column}_mean6"] = series.rolling(6, min_periods=3).mean(
            ).reindex(index).to_numpy()
            x[f"weather_{column}_mean24"] = series.rolling(24, min_periods=12).mean(
            ).reindex(index).to_numpy()
        if enhanced:
            extra = {}
            for column in w.columns:
                for window in (48, 168):
                    extra[f"weather_{column}_mean{window}"] = weather[column].rolling(
                        window, min_periods=window // 2).mean().reindex(index).to_numpy()
            x = pd.concat([x, pd.DataFrame(extra, index=index)], axis=1)
    return x.astype("float32")
