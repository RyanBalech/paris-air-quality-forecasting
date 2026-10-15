from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("holidays")

from paris_air_quality.advanced import advanced_forecast, features
from paris_air_quality.data import TARGETS


def fixture_frames():
    index = pd.date_range("2023-12-01", periods=3000, freq="h")
    targets = pd.DataFrame({t: 10 + np.sin(np.arange(3000) / 24) for t in TARGETS},
                           index=index)
    columns = ("temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m",
               "shortwave_radiation", "wind_direction_10m", "wind_speed_100m",
               "wind_direction_100m", "dew_point_2m", "direct_radiation")
    weather = pd.DataFrame({c: np.sin(np.arange(3000) / 48) + 20 for c in columns}, index=index)
    weather["boundary_layer_height"] = np.nan
    return targets, weather


@pytest.mark.parametrize("week", [1, 2, 3])
def test_week_specific_features_cannot_read_after_origin(week):
    targets, weather = fixture_frames()
    history = targets.iloc[:2496]
    future = targets.index[2496 + (week - 1) * 168:2496 + week * 168]
    clean = features(future, history, weather, lag_floor=week * 168)
    targets.loc[targets.index[2496:], :] = 1e9
    poisoned = features(future, targets, weather, lag_floor=week * 168)
    pd.testing.assert_frame_equal(clean, poisoned)
    assert not clean.columns.str.contains("boundary_layer_height").any()
    assert clean.iloc[-1][f"valeur_NO2_lag_{week * 168}"] == pytest.approx(
        history.iloc[-1]["valeur_NO2"])


@pytest.mark.parametrize("family,weekly", [("lightgbm", True), ("catboost", False)])
def test_alternative_forecasts_complete_horizon(family, weekly):
    if family == "catboost":
        pytest.importorskip("catboost")
    targets, weather = fixture_frames()
    prediction = advanced_forecast(targets.iloc[:2496], targets.index[2496:], weather,
                                   family=family, weekly=weekly, iterations=2)
    assert prediction.shape == (504, 5)
    assert prediction.index.equals(targets.index[2496:])
    assert np.isfinite(prediction.to_numpy()).all()
    assert (prediction >= 0).all().all()
    with pytest.raises(ValueError, match="contiguous"):
        advanced_forecast(targets.iloc[:2496], targets.index[2496:][::-1], weather)
