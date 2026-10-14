"""Fetch permitted weather reanalysis; never fetch pollutant observations."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

WEATHER_COLUMNS = (
    "temperature_2m", "relative_humidity_2m", "precipitation", "pressure_msl",
    "wind_speed_10m", "wind_direction_10m", "shortwave_radiation", "cloud_cover",
)
DEFAULT_CACHE = Path("data/cache/weather_era5.csv.gz")
API = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather(start: str = "2020-01-01", end: str = "2024-09-24",
                  output: Path = DEFAULT_CACHE) -> pd.DataFrame:
    """Download Paris ERA5 weather by calendar year and cache the exact response table."""
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    for year in range(first.year, last.year + 1):
        lo = max(first, pd.Timestamp(year=year, month=1, day=1))
        hi = min(last, pd.Timestamp(year=year, month=12, day=31))
        params = {"latitude": 48.8566, "longitude": 2.3522,
                  "start_date": lo.date().isoformat(), "end_date": hi.date().isoformat(),
                  "hourly": ",".join(WEATHER_COLUMNS), "timezone": "GMT", "models": "era5"}
        url = API + "?" + urlencode(params)
        print(f"Fetching ERA5 weather {year}: {lo.date()} to {hi.date()}", file=sys.stderr)
        with urlopen(url, timeout=120) as response:
            payload = json.load(response)
        if "hourly" not in payload or payload.get("error"):
            raise ValueError(f"Weather API failed for {year}: {payload.get('reason')}")
        frame = pd.DataFrame(payload["hourly"])
        frame.index = pd.to_datetime(frame.pop("time"), format="%Y-%m-%dT%H:%M")
        frames.append(frame)
    weather = pd.concat(frames).sort_index()
    if weather.index.has_duplicates or not weather.index.equals(
            pd.date_range(weather.index[0], weather.index[-1], freq="h")):
        raise ValueError("Weather timestamps are not unique and hourly-contiguous")
    if not set(WEATHER_COLUMNS).issubset(weather.columns):
        raise ValueError("Weather API omitted requested variables")
    output.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(output, index_label="time", compression="gzip")
    output.with_suffix(".json").write_text(json.dumps({
        "source": API, "model": "ERA5", "latitude": 48.8566, "longitude": 2.3522,
        "timezone": "GMT", "columns": WEATHER_COLUMNS,
        "start": start, "end": end,
    }, indent=2), encoding="utf-8")
    return weather


def load_weather(path: Path = DEFAULT_CACHE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Weather cache not found: {path}. Run `paris-air fetch-weather`.")
    return pd.read_csv(path, index_col="time", parse_dates=["time"])
