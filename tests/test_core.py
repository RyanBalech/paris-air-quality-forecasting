from __future__ import annotations

from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from paris_air_quality.data import TARGETS, competition_mae, load_competition
from paris_air_quality.features import HORIZON_HOURS, TARGET_LAGS, build_features
from paris_air_quality.models import direct_lightgbm, seasonal_median


def test_competition_loader_checks_horizon_and_column_order(tmp_path):
    train_ids = pd.date_range("2024-01-01", periods=600, freq="h")
    test_ids = pd.date_range(train_ids[-1] + pd.Timedelta(hours=1), periods=504, freq="h")
    train = pd.DataFrame({"id": train_ids.strftime("%Y-%m-%d %H")})
    test = pd.DataFrame({"id": test_ids.strftime("%Y-%m-%d %H")})
    for target in TARGETS:
        train[target] = 1.0
    sample = test.copy()
    for target in TARGETS:
        sample[target] = 0.0
    path = tmp_path / "competition.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("train.csv", train.to_csv(index=False))
        archive.writestr("test.csv", test.to_csv(index=False))
        archive.writestr("sample_submission.csv", sample.to_csv(index=False))
    loaded_train, loaded_test, loaded_sample = load_competition(path)
    assert loaded_train.shape == (600, 5)
    assert len(loaded_test) == HORIZON_HOURS
    assert loaded_sample.id.equals(test.id)


def test_metric_is_equal_weight_mean_and_ignores_missing_labels():
    actual = pd.DataFrame({target: [1.0, np.nan] for target in TARGETS})
    predicted = pd.DataFrame({target: [3.0, 100.0] for target in TARGETS})
    metric = competition_mae(actual, predicted)
    assert metric["score"] == 2.0
    assert all(n == 1 for n in metric["observed_rows"].values())


def test_features_never_read_future_pollutant_values():
    index = pd.date_range("2023-01-01", periods=1500, freq="h")
    history = pd.DataFrame({target: np.arange(len(index), dtype=float) for target in TARGETS},
                           index=index)
    future = pd.date_range(index[-1] + pd.Timedelta(hours=1), periods=504, freq="h")
    x = build_features(future, history)
    assert min(TARGET_LAGS) >= HORIZON_HOURS
    assert x.iloc[0]["valeur_NO2_lag_504"] == len(index) - 504
    assert x.iloc[-1]["valeur_NO2_lag_504"] == len(index) - 1
    assert not x.filter(like="valeur_NO2_lag").isna().all().all()


def test_weather_features_use_only_weather_at_or_before_the_prediction_hour():
    index = pd.date_range("2024-01-01", periods=1600, freq="h")
    history = pd.DataFrame({target: np.arange(1000, dtype=float) for target in TARGETS},
                           index=index[:1000])
    weather = pd.DataFrame({column: np.arange(len(index), dtype=float)
                            for column in ("temperature_2m", "precipitation",
                                           "wind_speed_10m", "relative_humidity_2m",
                                           "shortwave_radiation", "wind_direction_10m")}, index=index)
    future = index[1000:1504]
    x = build_features(future, history, weather)
    assert x.iloc[0]["weather_temperature_2m"] == 1000
    assert x.iloc[0]["weather_temperature_2m_lag3"] == 997
    assert x.iloc[0]["weather_temperature_2m_lag24"] == 976
    assert x.iloc[-1]["valeur_NO2_lag_504"] == 999


def test_forecasts_have_full_nonnegative_horizon():
    index = pd.date_range("2023-01-01", periods=1500, freq="h")
    history = pd.DataFrame({target: 10 + np.sin(np.arange(len(index)) / 24)
                            for target in TARGETS}, index=index)
    future = pd.date_range(index[-1] + pd.Timedelta(hours=1), periods=504, freq="h")
    for prediction in (seasonal_median(history, future),
                       direct_lightgbm(history, future, n_estimators=10),
                       direct_lightgbm(history, future, n_estimators=10, enhanced=True)):
        assert prediction.shape == (504, 5)
        assert prediction.notna().all().all()
        assert (prediction >= 0).all().all()
    with pytest.raises(ValueError, match="504-hour"):
        direct_lightgbm(history, future[:100], n_estimators=10)


def test_enhanced_features_ignore_poisoned_future_labels():
    index = pd.date_range("2023-01-01", periods=2504, freq="h")
    values = pd.DataFrame({t: np.arange(len(index), dtype=float) for t in TARGETS},
                          index=index)
    future = index[2000:]
    clean = build_features(future, values.iloc[:2000], enhanced=True)
    values.loc[future, :] = 1e12
    poisoned = build_features(future, values, enhanced=True)
    pd.testing.assert_frame_equal(clean, poisoned)
    assert clean.iloc[-1]["valeur_NO2_past_mean24"] == np.arange(1976, 2000).mean()


def test_metric_rejects_missing_predictions_and_misaligned_rows():
    actual = pd.DataFrame({target: [1., 2.] for target in TARGETS})
    invalid = actual.copy()
    invalid.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        competition_mae(actual, invalid)
    with pytest.raises(ValueError, match="timestamps"):
        competition_mae(actual, actual.iloc[::-1])
