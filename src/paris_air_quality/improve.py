"""Reproducible model selection with cached predictions and reserved audit windows.

Run with python -m paris_air_quality.improve. Audit results never choose the winner.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DEFAULT_ZIP, TARGETS, competition_mae, load_competition
from .experiment import FOLDS
from .models import direct_lightgbm
from .weather import DEFAULT_CACHE, load_weather

AUDIT_FOLDS = ("2023-12-01 22", "2024-04-01 22", "2024-07-01 22")
CANDIDATES = {
    "baseline": {},
    "longer_boosting": {"n_estimators": 1200},
    "smoothed_history": {"enhanced": True, "n_estimators": 800},
    "regularized": {"enhanced": True, "n_estimators": 1200,
                    "num_leaves": 15, "min_child_samples": 120},
}


def run() -> None:
    train, test, sample = load_competition()
    weather = load_weather(DEFAULT_CACHE)
    source_files = ("features.py", "models.py", "improve.py")
    provenance = {
        "data_sha256": hashlib.sha256(DEFAULT_ZIP.read_bytes()).hexdigest(),
        "weather_sha256": hashlib.sha256(DEFAULT_CACHE.read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256(
            Path(__file__).with_name(name).read_bytes()).hexdigest() for name in source_files},
        "candidates": CANDIDATES,
        "development_folds": FOLDS, "audit_folds": AUDIT_FOLDS,
    }
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:16]
    cache = Path("outputs/selection") / fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    records = []

    def evaluate(name: str, cutoff_text: str, stage: str) -> dict:
        key = f"{name}_{cutoff_text.replace(' ', '_').replace(':', '-')}"
        score_path = cache / f"{key}.json"
        prediction_path = cache / f"{key}.csv"
        if score_path.exists():
            row = json.loads(score_path.read_text())
        else:
            cutoff = pd.Timestamp(cutoff_text)
            history = train.loc[:cutoff]
            actual = train.loc[cutoff + pd.Timedelta(hours=1):].iloc[:504]
            if len(actual) != 504:
                raise ValueError("Incomplete validation horizon")
            started = time.perf_counter()
            prediction = direct_lightgbm(history, actual.index, weather, **CANDIDATES[name])
            row = {"method": name, "cutoff": cutoff_text,
                   "fit_predict_seconds": time.perf_counter() - started,
                   **competition_mae(actual, prediction)}
            prediction.to_csv(prediction_path)
            score_path.write_text(json.dumps(row, indent=2))
        row["stage"] = stage
        records.append(row)
        print(f"{stage} {cutoff_text} {name}: {row['score']:.5f}", flush=True)
        return row

    for name in CANDIDATES:
        for cutoff in FOLDS:
            evaluate(name, cutoff, "development")
    means = {name: float(np.mean([r["score"] for r in records if r["method"] == name]))
             for name in CANDIDATES}
    winner = min(means, key=means.get)
    selection = {"selected": winner, "development_means": means, **provenance}
    (cache / "selection_before_audit.json").write_text(json.dumps(selection, indent=2))
    print(f"Locked winner before audit: {winner}", flush=True)
    for name in dict.fromkeys(("baseline", winner)):
        for cutoff in AUDIT_FOLDS:
            evaluate(name, cutoff, "audit")
    audit = {name: float(np.mean([r["score"] for r in records
                                 if r["stage"] == "audit" and r["method"] == name]))
             for name in dict.fromkeys(("baseline", winner))}
    report = {**selection, "audit_means": audit, "records": records,
              "limitations": ["Audit windows are historical, not an independent future test.",
                              "Only three audit windows; no leaderboard score.",
                              "ERA5 weather is assumed known across the forecast horizon."]}
    Path("results/improvement.json").write_text(json.dumps(report, indent=2))
    prediction = direct_lightgbm(train, test.index, weather, **CANDIDATES[winner])
    for target in TARGETS:
        sample[target] = prediction[target].to_numpy()
    if len(sample) != 504 or not np.isfinite(sample[list(TARGETS)].to_numpy()).all():
        raise ValueError("Invalid submission")
    sample.to_csv("outputs/submission_improved.csv", index=False)
    print(json.dumps({"development": means, "audit": audit, "winner": winner}), flush=True)


if __name__ == "__main__":
    run()
