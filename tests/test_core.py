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


def test_forecasts_have_full_nonnegative_horizon():
    index = pd.date_range("2023-01-01", periods=1500, freq="h")
    history = pd.DataFrame({target: 10 + np.sin(np.arange(len(index)) / 24)
                            for target in TARGETS}, index=index)
    future = pd.date_range(index[-1] + pd.Timedelta(hours=1), periods=504, freq="h")
    for prediction in (seasonal_median(history, future),
                       direct_lightgbm(history, future, n_estimators=10)):
        assert prediction.shape == (504, 5)
        assert prediction.notna().all().all()
        assert (prediction >= 0).all().all()
    with pytest.raises(ValueError, match="504-hour"):
        direct_lightgbm(history, future[:100], n_estimators=10)
