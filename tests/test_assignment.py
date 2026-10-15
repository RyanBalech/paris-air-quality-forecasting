import numpy as np
import pandas as pd
import pytest

from paris_air_quality.assignment import persistence, seasonal_naive, split_validation
from paris_air_quality.data import TARGETS


def _history(length=300):
    index = pd.date_range("2024-01-01", periods=length, freq="h")
    return pd.DataFrame({target: np.arange(length, dtype=float) for target in TARGETS},
                        index=index)


def test_seasonal_naive_repeats_only_pre_cutoff_week():
    history = _history()
    future = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=504, freq="h")
    prediction = seasonal_naive(history, future)
    assert len(prediction) == 504
    assert prediction.iloc[0, 0] == history.iloc[-168, 0]
    assert prediction.iloc[168, 0] == prediction.iloc[0, 0]
    assert prediction.iloc[-1, 0] == history.iloc[-1, 0]
    assert persistence(history, future).iloc[-1, 0] == history.iloc[-1, 0]


def test_seasonal_naive_rejects_noncontiguous_future():
    history = _history()
    future = pd.date_range(history.index[-1] + pd.Timedelta(hours=2), periods=504, freq="h")
    with pytest.raises(ValueError, match="immediately"):
        seasonal_naive(history, future)


def test_validation_is_last_504_hours():
    train = _history(1000)
    history, actual = split_validation(train)
    assert len(actual) == 504
    assert len(history) == 496
    assert actual.index[0] == history.index[-1] + pd.Timedelta(hours=1)
