"""Summarize paired audit errors from saved predictions without fitting models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TARGETS, load_competition


def run() -> None:
    report = json.loads(Path("results/improvement.json").read_text())
    keys = ("data_sha256", "weather_sha256", "source_sha256", "candidates",
            "development_folds", "audit_folds")
    provenance = {key: report[key] for key in keys}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:16]
    cache = Path("outputs/selection") / fingerprint
    train, _, _ = load_competition()
    details = []
    paired = []
    for cutoff in report["audit_folds"]:
        rows = [r for r in report["records"] if r["stage"] == "audit" and r["cutoff"] == cutoff]
        scores = {r["method"]: r["score"] for r in rows}
        paired.append(scores[report["selected"]] - scores["baseline"])
        for method in dict.fromkeys(("baseline", report["selected"])):
            key = f"{method}_{cutoff.replace(' ', '_').replace(':', '-')}"
            prediction = pd.read_csv(cache / f"{key}.csv", index_col=0, parse_dates=[0])
            actual = train.reindex(prediction.index)
            error = (actual - prediction).abs()
            for target in TARGETS:
                for week in range(3):
                    chunk = error[target].iloc[week * 168:(week + 1) * 168]
                    details.append({"cutoff": cutoff, "method": method, "target": target,
                                    "horizon_week": week + 1,
                                    "mae": float(chunk.mean()),
                                    "observed_rows": int(chunk.notna().sum())})
    rng = np.random.default_rng(42)
    bootstrap = rng.choice(paired, size=(10000, len(paired)), replace=True).mean(axis=1)
    output = {
        "selected": report["selected"],
        "audit_paired_score_deltas": paired,
        "mean_delta": float(np.mean(paired)),
        "descriptive_95pct_fold_bootstrap": np.quantile(bootstrap, [.025, .975]).tolist(),
        "caveat": "Only three historical windows; interval is descriptive and not strong evidence "
                  "of generalization. Negative delta favors the selected model.",
        "per_week_target": details,
    }
    Path("results/diagnostics.json").write_text(json.dumps(output, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in output.items() if k != "per_week_target"}, indent=2))


if __name__ == "__main__":
    run()
