"""Create the assignment's figures and concise, data-backed written report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paris_air_quality.assignment import TARGET_LIST, split_validation
from paris_air_quality.data import load_competition


def create_report(results_path: Path = Path("outputs/assignment/validation_results.json"),
                  destination: Path = Path("docs/assignment")) -> Path:
    report = json.loads(results_path.read_text(encoding="utf-8"))
    train, test, _ = load_competition()
    history, actual = split_validation(train)
    destination.mkdir(parents=True, exist_ok=True)
    plots = destination / "figures"
    plots.mkdir(exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    recent = history.iloc[-24 * 60:]
    for axis, target in zip(axes, TARGET_LIST):
        axis.plot(recent.index, recent[target], lw=0.65)
        axis.set_ylabel(target.replace("valeur_", ""))
    axes[-1].set_xlabel("Date (last 60 pre-validation days)")
    fig.suptitle("Observed pollutant history before the forecast origin")
    fig.tight_layout()
    fig.savefig(plots / "pollutant_history.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    daily = history.groupby(history.index.hour).mean(numeric_only=True)
    weekly = history.groupby(history.index.dayofweek).mean(numeric_only=True)
    for target in TARGET_LIST:
        z = history[target].std()
        mean = history[target].mean()
        axes[0].plot(daily.index, (daily[target] - mean) / z, label=target.replace("valeur_", ""))
        axes[1].plot(weekly.index, (weekly[target] - mean) / z, label=target.replace("valeur_", ""))
    axes[0].set(xlabel="Hour of day", ylabel="Deviation in historical standard deviations",
                title="Mean hourly pattern")
    axes[1].set(xlabel="Day of week (Monday=0)", title="Mean weekly pattern")
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plots / "seasonality.png", dpi=150)
    plt.close(fig)

    correlations = history.corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    heat = ax.imshow(correlations.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(5), [t.replace("valeur_", "") for t in TARGET_LIST])
    ax.set_yticks(range(5), [t.replace("valeur_", "") for t in TARGET_LIST])
    fig.colorbar(heat, ax=ax)
    fig.tight_layout()
    fig.savefig(plots / "correlations.png", dpi=150)
    plt.close(fig)

    predictions = {}
    for method in report["methods"]:
        prediction = pd.read_csv(results_path.parent / f"validation_{method}.csv",
                                 index_col="timestamp", parse_dates=True)
        if not prediction.index.equals(actual.index):
            raise ValueError(f"Saved timestamps differ for {method}")
        predictions[method] = prediction
    selected = report["selected_method"]
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    for axis, target in zip(axes, TARGET_LIST):
        axis.plot(actual.index, actual[target], label="Observed", lw=0.8, color="black")
        axis.plot(actual.index, predictions["seasonal_naive_168h"][target],
                  label="Seasonal naive", lw=0.8, alpha=0.75)
        if selected != "seasonal_naive_168h":
            axis.plot(actual.index, predictions[selected][target], label=selected,
                      lw=0.8, alpha=0.85)
        axis.set_ylabel(target.replace("valeur_", ""))
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Validation timestamp")
    fig.suptitle("Observed and predicted values across the entire 504-hour horizon")
    fig.tight_layout()
    fig.savefig(plots / "observed_vs_predicted.png", dpi=150)
    plt.close(fig)

    stats = history.describe(percentiles=[0.01, 0.5, 0.99]).T
    rows = ["| Pollutant | Observed | Missing | Mean | Median | 1st–99th percentile | Min–max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for target in TARGET_LIST:
        s = stats.loc[target]
        fmt = ".3f" if target == "valeur_CO" else ".1f"
        rows.append(f"| {target} | {int(s['count']):,} | {int(history[target].isna().sum()):,} "
                    f"| {s['mean']:{fmt}} | {s['50%']:{fmt}} | "
                    f"{s['1%']:{fmt}}–{s['99%']:{fmt}} "
                    f"| {s['min']:{fmt}}–{s['max']:{fmt}} |")
    result_rows = ["| Method | NO2 | CO | O3 | PM10 | PM2.5 | Mean MAE | Runtime |",
                   "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method, outcome in sorted(report["methods"].items(),
                                  key=lambda item: item[1]["validation"]["score"]):
        scores = outcome["validation"]["per_target"]
        result_rows.append("| " + method + " | " + " | ".join(
            f"{scores[target]:.3f}" for target in TARGET_LIST) +
            f" | **{outcome['validation']['score']:.3f}** | {outcome['elapsed_seconds']:.1f}s |")
    best_each = {target: min(report["methods"], key=lambda method:
                             report["methods"][method]["validation"]["per_target"][target])
                 for target in TARGET_LIST}
    week_rows = ["| Method | Week 1 | Week 2 | Week 3 |",
                 "| --- | ---: | ---: | ---: |"]
    for method, outcome in report["methods"].items():
        week = outcome["validation"]["week_mae"]
        week_rows.append(f"| {method} | {week['1']['score']:.3f} | "
                         f"{week['2']['score']:.3f} | {week['3']['score']:.3f} |")
    chronos_failures = []
    for method in ("chronos2_independent", "chronos2_joint"):
        if method in report["methods"]:
            worse = [target for target in TARGET_LIST if
                     report["methods"][method]["validation"]["per_target"][target] >
                     report["methods"]["seasonal_naive_168h"]["validation"]["per_target"][target]]
            chronos_failures.append(f"{method}: {', '.join(worse) if worse else 'none'}")
    final_exists = (results_path.parent / "assignment_submission.csv").exists()
    chronos_settings = report["methods"]["chronos2_independent"]["settings"]
    peak_time = actual["valeur_O3"].idxmax()
    peak_actual = actual.loc[peak_time, "valeur_O3"]
    peak_predicted = predictions["chronos2_independent"].loc[peak_time, "valeur_O3"]
    lines = [
        "# Paris air-quality forecasting: assignment report", "",
        "## Task and data", "",
        (f"Forecast five pollutants for {len(test)} consecutive hourly test timestamps. "
        f"Training covers {train.index[0]} to {train.index[-1]}; the holdout is "
        f"{actual.index[0]} to {actual.index[-1]}. Timestamps come from the `id` column. "
        "The loader verifies ascending order, unique timestamps and contiguous hourly rows."), "",
        *rows, "",
        ("Values outside the central 1st–99th percentile range are inspected as possible extremes, "
        "not deleted automatically. Pollution peaks can be real; CO has a particularly long missing history. "
        "O3 includes a small negative value, so blind nonnegative clipping could change the benchmark."), "",
        "![Historical series](figures/pollutant_history.png)", "",
        "![Hourly and weekly patterns](figures/seasonality.png)", "",
        "![Correlations](figures/correlations.png)", "",
        ("The hourly averages show a daily pattern and the weekday averages show a smaller weekly "
        "pattern. The raw series contains sharp peaks and level changes. NO2 and CO are positively "
        "associated; O3 is negatively associated with them. Correlations are descriptive, not proof "
        "that one pollutant causes another."), "",
        "## Validation and leakage control", "",
        ("This is forecasting because target values after the forecast origin are unavailable at prediction "
        "time and errors can propagate through a 21-day horizon. A random row split mixes earlier and "
        "later hours and places near-duplicate seasonal observations on both sides; it would overstate "
        "future forecasting skill. The final 504 training hours form one contiguous validation window. "
        "Every method sees only observations through the cutoff. Missing inputs are filled by forward "
        "fill within that history, with a historical median fallback. Interpolating across validation or "
        "test would use unseen future pollutant measurements. No future weather observations are used."), "",
        ("Persistence repeats the last available value; the seasonal naïve method repeats the last "
        "168-hour week three times. A 168-hour lag preserves the hour of day and weekday, relevant "
        "to traffic and activity cycles. Scores omit missing validation labels per target, and then "
        "average the five target MAEs equally. Per-target MAE is essential because pollutant scales "
        "and dynamics differ; the mean can hide a regression for an individual pollutant."), "",
        "## Models and results", "",
        ("Chronos-2 is a pretrained time-series foundation model: its representations come from broad "
        "pretraining rather than fitting on this Paris dataset. Zero-shot means the downloaded model "
        "weights are used without fine-tuning here. Independent mode treats each pollutant as a separate "
        "series; joint mode gives all five to a single multivariate call. Joint forecasting can use "
        "their cross-series relationships, but can suffer from mismatched scales or correlations that "
        "change over time. The point forecast is the 0.5 quantile (median), which minimizes expected "
        "absolute error. The comparison LSTM is trained only on pre-cutoff pollutant history and "
        "recursively predicts 24-hour blocks."), "",
        (f"Chronos-2 package {chronos_settings['chronos']}, checkpoint revision "
        f"`{chronos_settings['model_revision']}`, 2,048 context hours, 504 output hours, "
        f"quantile 0.5, batch size 8. Hardware: {chronos_settings['gpu']}; "
        f"PyTorch {chronos_settings['torch']}. Independent inference took "
        f"{chronos_settings['inference_seconds']:.2f}s after model loading and used "
        f"{chronos_settings['gpu_peak_mb']:.0f} MiB peak allocated GPU memory. "
        "The runtime table includes import, checkpoint loading and forecasting; the first "
        "Chronos method pays cold-start costs, so those two wall times are not directly comparable."), "",
        *result_rows, "",
        f"The lowest mean MAE is **{selected}**. Best by pollutant: " +
        "; ".join(f"{target.replace('valeur_', '')}: {method}" for target, method in best_each.items()) +
        ". This demonstrates why a single average is insufficient.", "",
        "### Horizon behavior", "", *week_rows, "",
        "![Observed versus predicted](figures/observed_vs_predicted.png)", "",
        (f"A concrete failure is the O3 peak at {peak_time}: observed {peak_actual:.1f}, "
         f"independent Chronos-2 predicted {peak_predicted:.1f}. The weekly errors above show "
         "whether forecasts deteriorate between the beginning and end of the horizon. "
         "Specific Chronos-2 losses to the 168-hour seasonal baseline by pollutant: ") +
        "; ".join(chronos_failures) + ".", "",
        "## Covariates, limits and final forecast", "",
        ("Pollutant history is the target history. Historical weather is a past-only covariate. "
        "Hour of day is known in the future. Future *observed* weather is not known at the forecast "
        "origin and is prohibited here; a separately issued weather forecast would be a different "
        "future-available input, but none is used. The data limitation is long gaps and possible "
        "sensor changes; the evaluation limitation is one 21-day holdout, which has high seasonal "
        "uncertainty; the model limitation is that a zero-shot model is not adapted to Paris-specific "
        "events, while the small LSTM compounds recursive errors."), "",
        f"The final method was selected solely by validation mean MAE: **{selected}**. "
        + ("Its full-history test forecast is in `outputs/assignment/assignment_submission.csv`; "
           "the export checks all 504 rows, exact ID order, five target columns and finite values."
           if final_exists else "The final test forecast is pending model execution."), "",
        ("The foundation-model experiment was executed on the local machine and the notebook is "
        "Colab-ready. A Google Colab session itself is not claimed unless rerun there."), "",
        "## References", "",
        ("- Ansari et al., [Chronos-2: From Univariate to Universal Forecasting]"
        "(https://arxiv.org/abs/2510.15821)."),
        ("- Amazon Science, [Chronos-2 repository and quick start]"
        "(https://github.com/amazon-science/chronos-forecasting)."),
        ("- [X/HEC Paris air-quality Kaggle competition]"
        "(https://www.kaggle.com/competitions/x-hec-ts-2025-predicting-air-quality-in-paris/overview)."),
    ]
    path = destination / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(create_report())
