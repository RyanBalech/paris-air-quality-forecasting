"""Forecast-origin-safe experiments for the course assignment.

This workflow deliberately excludes the competition's archived future weather.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from paris_air_quality.data import TARGETS, competition_mae, load_competition

HORIZON = 504
TARGET_LIST = list(TARGETS)
CHRONOS_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


def history_for_model(history: pd.DataFrame) -> pd.DataFrame:
    """Fill only observations available at the forecast origin."""
    values = history.loc[:, TARGET_LIST].astype("float64")
    medians = values.median().fillna(0.0)
    return values.ffill().fillna(medians)


def persistence(history: pd.DataFrame, future_index: pd.DatetimeIndex) -> pd.DataFrame:
    last = history_for_model(history).iloc[-1].to_numpy()
    return pd.DataFrame(np.tile(last, (len(future_index), 1)), index=future_index,
                        columns=TARGET_LIST)


def seasonal_naive(history: pd.DataFrame, future_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Repeat the last observed 168-hour week across the full horizon."""
    if len(history) < 168:
        raise ValueError("Seasonal naive needs at least 168 hours of history")
    if future_index[0] != history.index[-1] + pd.Timedelta(hours=1):
        raise ValueError("Forecast must start immediately after history")
    week = history_for_model(history).iloc[-168:].to_numpy()
    return pd.DataFrame(week[np.arange(len(future_index)) % 168],
                        index=future_index, columns=TARGET_LIST)


def split_validation(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(train) <= HORIZON + 168:
        raise ValueError("Not enough history for a 504-hour validation window")
    history, actual = train.iloc[:-HORIZON].copy(), train.iloc[-HORIZON:].copy()
    if actual.index[0] != history.index[-1] + pd.Timedelta(hours=1):
        raise ValueError("Validation period is not contiguous")
    return history, actual


def score(actual: pd.DataFrame, prediction: pd.DataFrame) -> dict:
    result = competition_mae(actual, prediction)
    result["week_mae"] = {
        str(week + 1): competition_mae(actual.iloc[week * 168:(week + 1) * 168],
                                       prediction.iloc[week * 168:(week + 1) * 168])
        for week in range(3)
    }
    return result


def lstm_forecast(history: pd.DataFrame, future_index: pd.DatetimeIndex,
                  epochs: int = 8, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Fit a small joint LSTM for 24-hour steps, then roll forward 21 days."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    observed = history_for_model(history)
    # Recent data reduces obsolete sensor regimes and keeps a laptop run tractable.
    values = observed.iloc[-24 * 365 * 2:].to_numpy(dtype=np.float32)
    center = np.median(values, axis=0)
    scale = np.std(values, axis=0).clip(min=1e-3)
    normalized = (values - center) / scale
    context, step, stride = 168, 24, 12
    starts = np.arange(0, len(normalized) - context - step + 1, stride)
    x = torch.from_numpy(np.stack([normalized[i:i + context] for i in starts]))
    y = torch.from_numpy(np.stack([normalized[i + context:i + context + step]
                                  for i in starts]))

    class ForecastLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.recurrent = nn.LSTM(len(TARGETS), 64, num_layers=1, batch_first=True)
            self.head = nn.Linear(64, step * len(TARGETS))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            _, (hidden, _) = self.recurrent(inputs)
            return self.head(hidden[-1]).reshape(-1, step, len(TARGETS))

    model = ForecastLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(x))
        batch_losses = []
        for ids in permutation.split(128):
            prediction = model(x[ids].to(device))
            loss = nn.functional.l1_loss(prediction, y[ids].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(batch_losses)))

    model.eval()
    rolling = normalized[-context:].copy()
    blocks = []
    with torch.inference_mode():
        for _ in range(int(np.ceil(len(future_index) / step))):
            block = model(torch.from_numpy(rolling[-context:][None]).to(device))
            output = block[0].cpu().numpy()
            blocks.append(output)
            rolling = np.concatenate([rolling, output], axis=0)
    forecast = np.concatenate(blocks, axis=0)[:len(future_index)] * scale + center
    metadata = {"device": device,
                "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                "torch": torch.__version__, "epochs": epochs,
                "seed": seed, "training_windows": len(starts), "training_losses": losses,
                "context_length": context, "block_length": step, "stride": stride,
                "training_hours": len(values),
                "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None}
    return pd.DataFrame(forecast, index=future_index, columns=TARGET_LIST), metadata


def chronos_forecast(history: pd.DataFrame, future_index: pd.DatetimeIndex,
                     joint: bool, context_length: int = 2048) -> tuple[pd.DataFrame, dict]:
    """Run the pretrained Chronos-2 checkpoint without fitting on Paris data."""
    import torch
    from chronos import Chronos2Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2", revision=CHRONOS_REVISION, device_map=device)
    load_seconds = time.perf_counter() - load_started
    context = history_for_model(history).iloc[-context_length:].copy()
    context.index.name = "timestamp"
    context = context.reset_index()
    context["item_id"] = "paris"
    target = TARGET_LIST if joint else "target"
    if not joint:
        context = context.melt(id_vars=["timestamp", "item_id"], value_vars=TARGET_LIST,
                               var_name="pollutant", value_name="target")
        context["item_id"] = context["pollutant"]
        context = context.drop(columns="pollutant").sort_values(["item_id", "timestamp"])
    inference_started = time.perf_counter()
    prediction = pipeline.predict_df(
        context, id_column="item_id", timestamp_column="timestamp", target=target,
        prediction_length=len(future_index), quantile_levels=[0.5],
        context_length=context_length, freq="h", batch_size=8,
        unrolled_quantiles=[0.5],
    )
    if device == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    # Chronos-2 returns long output for a list of targets; a scalar target returns
    # one series per item_id. The 0.5 quantile is the MAE-optimal point forecast.
    forecast = pd.DataFrame(index=future_index, columns=TARGET_LIST, dtype="float64")
    prediction["timestamp"] = pd.to_datetime(prediction["timestamp"])
    for pollutant in TARGET_LIST:
        if joint:
            section = prediction.loc[prediction["target_name"] == pollutant]
        else:
            section = prediction.loc[prediction["item_id"] == pollutant]
        if len(section) != len(future_index):
            raise ValueError(f"Chronos returned {len(section)} rows for {pollutant}")
        series = section.set_index("timestamp")["0.5"].astype(float)
        if not series.index.equals(future_index):
            raise ValueError(f"Chronos timestamps do not match for {pollutant}")
        forecast[pollutant] = series
    metadata = {"device": device,
                "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                "torch": torch.__version__,
                "chronos": version("chronos-forecasting"),
                "model": "amazon/chronos-2", "context_length": len(context) if joint else context_length,
                "quantile": 0.5, "joint": joint, "batch_size": 8,
                "model_revision": CHRONOS_REVISION,
                "model_native_horizon": pipeline.model_prediction_length,
                "load_seconds": load_seconds, "inference_seconds": inference_seconds,
                "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None}
    return forecast, metadata


def _save_csv(prediction: pd.DataFrame, sample: pd.DataFrame, output: Path) -> None:
    result = sample.copy()
    if len(prediction) != len(result) or not np.isfinite(prediction.to_numpy()).all():
        raise ValueError("Incomplete or nonfinite test forecast")
    result.loc[:, TARGET_LIST] = prediction.to_numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    check = pd.read_csv(output)
    if list(check.columns) != list(sample.columns) or not check.id.equals(sample.id):
        raise ValueError("Submission format changed while saving")


def run(output_dir: Path = Path("outputs/assignment"), epochs: int = 8,
        skip_chronos: bool = False) -> dict:
    train, test, sample = load_competition()
    history, actual = split_validation(train)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = {
        "persistence": lambda h, i: (persistence(h, i), {}),
        "seasonal_naive_168h": lambda h, i: (seasonal_naive(h, i), {}),
        "lstm_joint": lambda h, i: lstm_forecast(h, i, epochs=epochs),
    }
    if not skip_chronos:
        methods.update({
            "chronos2_independent": lambda h, i: chronos_forecast(h, i, joint=False),
            "chronos2_joint": lambda h, i: chronos_forecast(h, i, joint=True),
        })
    report = {
        "protocol": {"validation_start": str(actual.index[0]),
                     "validation_end": str(actual.index[-1]), "forecast_hours": HORIZON,
                     "forecast_origin": str(history.index[-1]), "targets": TARGET_LIST,
                     "metric": "mean of five observed-label MAEs", "future_weather": False},
        "data": {"train_rows": len(train), "test_rows": len(test),
                 "missing_train": train.isna().sum().to_dict(),
                 "missing_validation": actual.isna().sum().to_dict(),
                 "train_sha256": hashlib.sha256(
                     Path("data/raw/competition.zip").read_bytes()).hexdigest()},
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
        "methods": {}, "selected_method": None,
    }
    results_path = output_dir / "validation_results.json"
    if results_path.exists():
        previous = json.loads(results_path.read_text(encoding="utf-8"))
        if previous.get("protocol") != report["protocol"] or previous.get("data") != report["data"]:
            raise ValueError("Existing checkpoint uses different data or validation settings")
        report["methods"] = previous.get("methods", {})
    for name, predict in methods.items():
        forecast_path = output_dir / f"validation_{name}.csv"
        if name in report["methods"] and forecast_path.exists():
            cached = pd.read_csv(forecast_path, index_col="timestamp", parse_dates=True)
            complete = not name.startswith("chronos2_") or "inference_seconds" in (
                report["methods"][name].get("settings", {}))
            if complete and cached.index.equals(actual.index) and np.isfinite(
                    cached[TARGET_LIST].to_numpy()).all():
                continue
        started = time.perf_counter()
        forecast, metadata = predict(history, actual.index)
        elapsed = time.perf_counter() - started
        forecast.to_csv(forecast_path, index_label="timestamp")
        report["methods"][name] = {"validation": score(actual, forecast),
                                   "elapsed_seconds": elapsed, "settings": metadata}
        results_path.write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
    selected = min(report["methods"], key=lambda name:
                   report["methods"][name]["validation"]["score"])
    report["selected_method"] = selected
    started = time.perf_counter()
    final, metadata = methods[selected](train, test.index)
    report["final_fit_seconds"] = time.perf_counter() - started
    report["final_settings"] = metadata
    _save_csv(final, sample, output_dir / "assignment_submission.csv")
    results_path.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/assignment"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--skip-chronos", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.epochs, args.skip_chronos), indent=2))


if __name__ == "__main__":
    main()
