"""Create audit diagnostics and a shareable comparison figure from completed runs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import TARGETS


def run() -> None:
    report = json.loads(Path("results/deep_study.json").read_text())
    rows = [r for r in report["records"] if r["stage"] == "audit"]
    paired = []
    table = []
    for cutoff in report["audit"]:
        baseline = next(r for r in rows if r["method"] == "baseline" and r["cutoff"] == cutoff)
        selected = next(r for r in rows if r["method"] == "selected_blend" and r["cutoff"] == cutoff)
        paired.append(selected["score"] - baseline["score"])
        table.append({"cutoff": cutoff, "baseline": baseline["score"],
                      "selected": selected["score"], "delta": paired[-1]})
    rng = np.random.default_rng(42)
    samples = rng.choice(paired, size=(10000, len(paired)), replace=True).mean(axis=1)
    diagnostic = {
        "folds": table, "mean_delta": float(np.mean(paired)),
        "descriptive_95pct_fold_bootstrap": np.quantile(samples, [.025, .975]).tolist(),
        "caveat": "Only three audit windows; historical reuse and model selection limit inference.",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(table))
    axes[0].bar(x - .18, [r["baseline"] for r in table], .36, label="Baseline", color="#64748b")
    axes[0].bar(x + .18, [r["selected"] for r in table], .36, label="Selected blend", color="#0284c7")
    axes[0].set_xticks(x, [r["cutoff"][:10] for r in table])
    axes[0].set_ylabel("Mean pollutant MAE (lower is better)")
    axes[0].set_title("Reserved historical audit windows")
    axes[0].legend(frameon=False)
    baseline = report["summary"]["audit"]["baseline"]["per_target"]
    selected = report["summary"]["audit"]["selected_blend"]["per_target"]
    gain = [100 * (1 - selected[t] / baseline[t]) for t in TARGETS]
    axes[1].bar([t.removeprefix("valeur_") for t in TARGETS], gain,
                color=["#0284c7" if g >= 0 else "#dc2626" for g in gain])
    axes[1].axhline(0, color="#334155", linewidth=.8)
    axes[1].set_ylabel("MAE reduction (%)")
    axes[1].set_title("Audit change by pollutant")
    fig.suptitle("Paris air-quality forecasts — local validation, no Kaggle score", fontsize=12)
    fig.tight_layout()
    Path("docs/figures").mkdir(exist_ok=True)
    fig.savefig("docs/figures/deep_audit.png", dpi=160)
    plt.close(fig)
    Path("results/deep_diagnostics.json").write_text(json.dumps(diagnostic, indent=2))
    print(json.dumps(diagnostic, indent=2))


if __name__ == "__main__":
    run()
