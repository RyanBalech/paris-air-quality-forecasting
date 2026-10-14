"""Command-line entry points for data preparation, backtests, and submissions."""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import DEFAULT_ZIP
from .experiment import METHODS, backtest, submission
from .weather import DEFAULT_CACHE, fetch_weather


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast Paris air quality")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-weather")
    fetch.add_argument("--output", type=Path, default=DEFAULT_CACHE)
    evaluate = commands.add_parser("backtest")
    evaluate.add_argument("--data", type=Path, default=DEFAULT_ZIP)
    evaluate.add_argument("--weather", type=Path, default=DEFAULT_CACHE)
    evaluate.add_argument("--output", type=Path, default=Path("outputs/backtest.json"))
    predict = commands.add_parser("submit")
    predict.add_argument("--data", type=Path, default=DEFAULT_ZIP)
    predict.add_argument("--weather", type=Path, default=DEFAULT_CACHE)
    predict.add_argument("--method", choices=METHODS, default="lightgbm_weather")
    predict.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
    args = parser.parse_args()
    if args.command == "fetch-weather":
        weather = fetch_weather(output=args.output)
        print(f"Cached {len(weather)} hourly weather rows at {args.output}")
    elif args.command == "backtest":
        report = backtest(args.data, args.weather, args.output)
        for method, row in report["summary"].items():
            print(f"{method}: mean score {row['mean_score']:.4f}")
        print(f"Saved {args.output}")
    else:
        print(f"Saved {submission(args.data, args.weather, args.method, args.output)}")


if __name__ == "__main__":
    main()
