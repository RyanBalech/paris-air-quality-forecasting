"""Refit a previously selected blend without rerunning model selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .advanced_weather import fetch_extended
from .data import TARGETS, load_competition
from .deep_study import CONFIGS, blend, predict
from .weather import load_weather


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("results/deep_study.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submission_deep.csv"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report["configs"] != CONFIGS:
        raise ValueError("Saved model configurations differ from this source version")
    train, test, sample = load_competition()
    weather, extra = load_weather(), fetch_extended()
    recipe = report["recipe"]
    members = sorted({name for names in recipe.values() for name in names})
    predictions = {name: predict(name, train, test.index, weather, extra) for name in members}
    forecast = blend(predictions, recipe)
    if not forecast.index.equals(test.index) or not np.isfinite(forecast.to_numpy()).all():
        raise ValueError("Invalid forecast")
    for target in TARGETS:
        sample[target] = forecast[target].to_numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
