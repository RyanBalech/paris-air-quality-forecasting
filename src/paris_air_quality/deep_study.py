"""Compare alternative inductive biases and lock blends before fresh historical audits."""
from __future__ import annotations

import hashlib
import itertools
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .advanced import advanced_forecast
from .advanced_weather import EXTENDED_CACHE, fetch_extended
from .data import DEFAULT_ZIP, TARGETS, competition_mae, load_competition
from .experiment import FOLDS
from .models import direct_lightgbm
from .weather import DEFAULT_CACHE, load_weather

AUDIT = ("2023-10-01 22", "2024-01-01 22", "2024-05-01 22")
CONFIGS = {
    "baseline": {},
    "regularized": {"enhanced": True, "n_estimators": 1200,
                    "num_leaves": 15, "min_child_samples": 120},
    "dispersion": {"family": "lightgbm"},
    "weather_only": {"family": "lightgbm", "use_history": False},
    "catboost": {"family": "catboost"},
    "weekly": {"family": "lightgbm", "weekly": True},
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(name: str, history: pd.DataFrame, future: pd.DatetimeIndex,
            weather: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    if name in ("baseline", "regularized"):
        return direct_lightgbm(history, future, weather, **CONFIGS[name])
    return advanced_forecast(history, future, weather.join(extra), **CONFIGS[name])


def blend(predictions: dict[str, pd.DataFrame], recipe: dict) -> pd.DataFrame:
    output = pd.DataFrame(index=next(iter(predictions.values())).index)
    for target, members in recipe.items():
        output[target] = sum(predictions[m][target] for m in members) / len(members)
    return output.loc[:, list(TARGETS)]


def run() -> None:
    train, test, sample = load_competition()
    weather = load_weather()
    extra = fetch_extended()
    prior = json.loads(Path("results/improvement.json").read_text())
    old_keys = ("data_sha256", "weather_sha256", "source_sha256", "candidates",
                "development_folds", "audit_folds")
    old_fingerprint = hashlib.sha256(json.dumps(
        {k: prior[k] for k in old_keys}, sort_keys=True).encode()).hexdigest()[:16]
    old_cache = Path("outputs/selection") / old_fingerprint
    source = Path(__file__).parent
    can_reuse = (prior["data_sha256"] == digest(DEFAULT_ZIP)
                 and prior["weather_sha256"] == digest(DEFAULT_CACHE)
                 and all(prior["source_sha256"][n] == digest(source / n)
                         for n in ("features.py", "models.py")))
    protocol = {
        "development": list(FOLDS), "audit": list(AUDIT), "configs": CONFIGS,
        "data_sha256": digest(DEFAULT_ZIP), "weather_sha256": digest(DEFAULT_CACHE),
        "extra_weather_sha256": digest(EXTENDED_CACHE),
        "source_sha256": {n: digest(source / n) for n in (
            "advanced.py", "features.py", "models.py", "data.py", "deep_study.py")},
        "software": {n: version(n) for n in (
            "numpy", "pandas", "lightgbm", "catboost", "holidays", "scikit-learn")},
        "selection": "Per-target lowest development MAE among singles and equal-weight pairs",
    }
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    cache = Path("outputs/deep_study") / fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "protocol.json").write_text(json.dumps(protocol, indent=2))
    records = []
    predictions = {}

    def evaluate(name: str, cutoff: str, stage: str) -> pd.DataFrame:
        key = f"{name}_{cutoff.replace(' ', '_')}"
        prediction_path = cache / f"{key}.csv"
        score_path = cache / f"{key}.json"
        actual = train.loc[pd.Timestamp(cutoff) + pd.Timedelta(hours=1):].iloc[:504]
        if len(actual) != 504:
            raise ValueError("Incomplete scoring horizon")
        old_path = old_cache / f"{key}.csv"
        if prediction_path.exists() and score_path.exists():
            prediction = pd.read_csv(prediction_path, index_col=0, parse_dates=[0])
            metadata = json.loads(score_path.read_text())
        elif can_reuse and name in ("baseline", "regularized") and old_path.exists():
            prediction = pd.read_csv(old_path, index_col=0, parse_dates=[0])
            metadata = {"reused_from": str(old_path), "seconds": None}
        else:
            started = time.perf_counter()
            prediction = predict(name, train.loc[:cutoff], actual.index, weather, extra)
            metadata = {"seconds": time.perf_counter() - started}
        metric = competition_mae(actual, prediction)
        row = {**metadata, "method": name, "cutoff": cutoff, "stage": stage, **metric,
               "weekly_scores": [competition_mae(actual.iloc[i:i+168],
                                                  prediction.iloc[i:i+168])["score"]
                                 for i in (0, 168, 336)]}
        prediction.to_csv(prediction_path)
        score_path.write_text(json.dumps(row, indent=2))
        records.append(row)
        predictions[name, cutoff] = prediction
        print(f"{stage} {cutoff} {name}: {metric['score']:.5f}", flush=True)
        return prediction

    for name in CONFIGS:
        for cutoff in FOLDS:
            evaluate(name, cutoff, "development")
    candidates = [(n,) for n in CONFIGS] + list(itertools.combinations(CONFIGS, 2))
    recipe = {}
    blend_scores = {}
    for target in TARGETS:
        scores = {}
        for members in candidates:
            fold_scores = []
            for cutoff in FOLDS:
                predicted = sum(predictions[m, cutoff][target] for m in members) / len(members)
                actual = train.loc[predicted.index, target]
                fold_scores.append(float((actual - predicted).abs().mean()))
            scores["+".join(members)] = float(np.mean(fold_scores))
        selected = min(scores, key=scores.get)
        recipe[target] = selected.split("+")
        blend_scores[target] = scores
    selection = {"recipe": recipe, "per_target_development_scores": blend_scores}
    (cache / "selection_before_audit.json").write_text(json.dumps(selection, indent=2))
    print("LOCKED RECIPE " + json.dumps(recipe), flush=True)
    needed = sorted({m for members in recipe.values() for m in members} | {"baseline"})
    for cutoff in AUDIT:
        for name in needed:
            evaluate(name, cutoff, "audit")
    for stage, cutoffs in (("development", FOLDS), ("audit", AUDIT)):
        for cutoff in cutoffs:
            combined = blend({n: predictions[n, cutoff] for n in needed}, recipe)
            actual = train.loc[combined.index]
            records.append({"method": "selected_blend", "cutoff": cutoff, "stage": stage,
                            **competition_mae(actual, combined),
                            "weekly_scores": [competition_mae(actual.iloc[i:i+168],
                                                               combined.iloc[i:i+168])["score"]
                                              for i in (0, 168, 336)]})
            combined.to_csv(cache / f"selected_blend_{cutoff.replace(' ', '_')}.csv")
    summary = {}
    for stage in ("development", "audit"):
        summary[stage] = {}
        for method in (*CONFIGS, "selected_blend"):
            rows = [r for r in records if r["method"] == method and r["stage"] == stage]
            if rows:
                summary[stage][method] = {
                    "score": float(np.mean([r["score"] for r in rows])),
                    "per_target": {t: float(np.mean([r["per_target"][t] for r in rows]))
                                   for t in TARGETS}}
    report = {**protocol, **selection, "cache": str(cache), "records": records,
              "summary": summary,
              "limitations": ["Three historical audit windows, not an untouched chronological test.",
                              "Candidate selection can overfit four development windows.",
                              "Test labels are partly imputed by organizers; local MAE uses observed labels.",
                              "Boundary-layer height excluded due to Jan-Jun 2024 API gap."]}
    Path("results/deep_study.json").write_text(json.dumps(report, indent=2))
    final = {}
    for name in needed:
        path = cache / f"final_{name}.csv"
        if path.exists():
            final[name] = pd.read_csv(path, index_col=0, parse_dates=[0])
        else:
            final[name] = predict(name, train, test.index, weather, extra)
            final[name].to_csv(path)
    forecast = blend(final, recipe)
    if not forecast.index.equals(test.index) or not np.isfinite(forecast.to_numpy()).all():
        raise ValueError("Invalid final predictions")
    for target in TARGETS:
        sample[target] = forecast[target].to_numpy()
    sample.to_csv("outputs/submission_deep.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()
