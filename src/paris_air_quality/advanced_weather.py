"""Additional permitted ERA5 weather predictors of atmospheric dispersion."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from .weather import API

EXTENDED_CACHE = Path("data/cache/weather_dispersion.csv.gz")
VARIABLES = ("boundary_layer_height", "wind_speed_100m", "wind_direction_100m",
             "dew_point_2m", "direct_radiation")


def fetch_extended() -> pd.DataFrame:
    if EXTENDED_CACHE.exists():
        return pd.read_csv(EXTENDED_CACHE, index_col="time", parse_dates=["time"])
    frames = []
    for year in range(2020, 2025):
        part = EXTENDED_CACHE.with_name(f"dispersion_{year}.csv.gz")
        if part.exists():
            frame = pd.read_csv(part, index_col="time", parse_dates=["time"])
        else:
            params = {"latitude": 48.8566, "longitude": 2.3522,
                      "start_date": f"{year}-01-01",
                      "end_date": f"{year}-12-31" if year < 2024 else "2024-09-24",
                      "hourly": ",".join(VARIABLES), "models": "era5", "timezone": "GMT"}
            with urlopen(API + "?" + urlencode(params), timeout=120) as response:
                payload = json.load(response)
            frame = pd.DataFrame(payload["hourly"])
            frame.index = pd.to_datetime(frame.pop("time"))
            part.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(part, index_label="time", compression="gzip")
        frames.append(frame)
        print(f"Dispersion weather ready: {year}", flush=True)
    result = pd.concat(frames)
    if result.index.has_duplicates or not result.index.equals(
            pd.date_range(result.index[0], result.index[-1], freq="h")):
        raise ValueError("Dispersion weather timestamps are not hourly-contiguous")
    if result.drop(columns="boundary_layer_height").isna().any().any():
        raise ValueError("Incomplete dispersion weather cache")
    result.to_csv(EXTENDED_CACHE, index_label="time", compression="gzip")
    return result


if __name__ == "__main__":
    fetch_extended()
