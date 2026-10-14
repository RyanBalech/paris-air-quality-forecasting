"""Compare training-history windows using the same 504-hour validation folds."""
import json
from pathlib import Path

import pandas as pd

from paris_air_quality.data import DEFAULT_ZIP, competition_mae, load_competition
from paris_air_quality.experiment import FOLDS
from paris_air_quality.features import HORIZON_HOURS
from paris_air_quality.models import direct_lightgbm
from paris_air_quality.weather import load_weather


def main():
    train, _, _ = load_competition(DEFAULT_ZIP)
    weather = load_weather()
    rows = []
    for years in (2, 4):
        for cutoff_text in FOLDS:
            cutoff = pd.Timestamp(cutoff_text)
            history = train.loc[:cutoff]
            actual = train.loc[cutoff + pd.Timedelta(hours=1):].iloc[:HORIZON_HOURS]
            prediction = direct_lightgbm(history, actual.index, weather, train_years=years)
            metric = competition_mae(actual, prediction)
            rows.append({"train_years": years, "cutoff": cutoff_text, **metric})
            print(f"{years} years, {cutoff_text}: {metric['score']:.4f}", flush=True)
    result = {"folds": rows, "means": {
        str(years): sum(r["score"] for r in rows if r["train_years"] == years) / len(FOLDS)
        for years in (2, 4)
    }}
    output = Path("outputs/train_window_probe.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result["means"])


if __name__ == "__main__":
    main()
