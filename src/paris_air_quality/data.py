"""Load and validate the competition's original ZIP without modifying its data."""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

TARGETS = ("valeur_NO2", "valeur_CO", "valeur_O3", "valeur_PM10", "valeur_PM25")
DEFAULT_ZIP = Path("data/raw/competition.zip")


def load_competition(path: Path = DEFAULT_ZIP) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with ZipFile(path) as archive:
        train = pd.read_csv(archive.open("train.csv"))
        test = pd.read_csv(archive.open("test.csv"))
        sample = pd.read_csv(archive.open("sample_submission.csv"))
    if list(train.columns) != ["id", *TARGETS] or list(test.columns) != ["id"]:
        raise ValueError("Unexpected training or test columns")
    if list(sample.columns) != ["id", *TARGETS] or not sample.id.equals(test.id):
        raise ValueError("Sample submission does not match test IDs")
    train.index = pd.to_datetime(train.pop("id"), format="%Y-%m-%d %H")
    test.index = pd.to_datetime(test.pop("id"), format="%Y-%m-%d %H")
    if train.index.has_duplicates or test.index.has_duplicates:
        raise ValueError("Duplicate timestamps in competition data")
    if not train.index.is_monotonic_increasing or not test.index.is_monotonic_increasing:
        raise ValueError("Competition timestamps are out of order")
    if test.index[0] != train.index[-1] + pd.Timedelta(hours=1):
        raise ValueError("Training and test windows are not contiguous")
    if not train.index.equals(pd.date_range(train.index[0], train.index[-1], freq="h")):
        raise ValueError("Training rows are not hourly-contiguous")
    if not test.index.equals(pd.date_range(test.index[0], test.index[-1], freq="h")):
        raise ValueError("Test rows are not hourly-contiguous")
    return train, test, sample


def competition_mae(actual: pd.DataFrame, predicted: pd.DataFrame) -> dict:
    """Mean of five per-target MAEs, scoring only observed validation labels."""
    if not actual.index.equals(predicted.index):
        raise ValueError("Prediction timestamps must match actual timestamps exactly")
    if not np.isfinite(predicted.loc[:, list(TARGETS)].to_numpy()).all():
        raise ValueError("Predictions must all be finite")
    scores = {}
    for target in TARGETS:
        mask = actual[target].notna()
        if not mask.any():
            raise ValueError(f"No observed labels for {target}")
        scores[target] = float((actual.loc[mask, target] - predicted.loc[mask, target]).abs().mean())
    return {"score": sum(scores.values()) / len(scores), "per_target": scores,
            "observed_rows": {t: int(actual[t].notna().sum()) for t in TARGETS}}
