"""Command-line interface for bac_forecast."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from .data import load_data
from .features import build_features
from .train import train_models, predict
from .evaluate import walk_forward_evaluate, print_report, save_results


def _train(args: argparse.Namespace) -> int:
    """Train models and save artifacts."""
    print(f"[train] Fetching data from {args.start} to {args.end} ...")
    raw = load_data(start=args.start, end=args.end)
    print(f"[train] Downloaded {len(raw)} rows.")

    print("[train] Engineering features ...")
    df = build_features(raw)
    print(f"[train] Feature matrix: {df.shape}")

    X = df.drop(columns=["Target"])
    y = df["Target"]

    split_idx = int(len(X) * (1 - args.test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"[train] Train: {len(X_train)} | Test: {len(X_test)}")

    print("[train] Training models ...")
    fitted = train_models(X_train, y_train)

    preds = predict(fitted, X_test)
    print("[train] Predictions generated.")

    # Save predictions CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.concat([y_test.rename("Actual"), preds], axis=1)
    result_df.to_csv(out_path)
    print(f"[train] Predictions saved to {out_path}")

    print("[train] Running walk-forward evaluation ...")
    results = walk_forward_evaluate(X, y, fitted, test_size=args.test_size)
    print_report(results)

    report_path = Path(args.output).parent / "reports" / "metrics.json"
    save_results(results, report_path)

    return 0


def _predict(args: argparse.Namespace) -> int:
    """Run prediction for a date range."""
    start = args.start or pd.Timestamp.today().strftime("%Y-%m-%d")
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"[predict] Fetching data from {start} to {end} ...")

    raw = load_data(start=start, end=end)

    print("[predict] Engineering features ...")
    df = build_features(raw)

    if df.empty:
        print("[predict] No valid data for the given range.")
        return 1

    X = df.drop(columns=["Target"])
    y = df["Target"]

    print("[predict] Training on available history ...")
    fitted = train_models(X, y)
    preds = predict(fitted, X)

    print("\n[predict] Model predictions:")
    result_df = pd.concat([y.rename("Actual"), preds], axis=1)
    print(result_df.to_string())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path)
    print(f"\n[predict] Saved to {out_path}")

    return 0


def _evaluate(args: argparse.Namespace) -> int:
    """Run full evaluation with walk-forward validation."""
    print("[evaluate] Loading data ...")
    raw = load_data(start=args.start, end=args.end)
    df = build_features(raw)

    X = df.drop(columns=["Target"])
    y = df["Target"]

    split_idx = int(len(X) * (1 - args.test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print("[evaluate] Training models ...")
    fitted = train_models(X_train, y_train)

    print("[evaluate] Running walk-forward evaluation ...")
    results = walk_forward_evaluate(X, y, fitted, test_size=args.test_size)
    print_report(results)

    report_path = Path(args.output)
    save_results(results, report_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bac-forecast",
        description="BAC next-day price forecasting with ML + macro indicators.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train models and save predictions.")
    p_train.add_argument(
        "--start", default="2002-01-01", help="Start date (YYYY-MM-DD)."
    )
    p_train.add_argument(
        "--end", default="2025-01-01", help="End date (YYYY-MM-DD)."
    )
    p_train.add_argument(
        "--test-size",
        type=float,
        default=0.10,
        help="Fraction of data for hold-out test (default: 0.10).",
    )
    p_train.add_argument(
        "--output",
        default="predictions.csv",
        help="Output CSV path.",
    )
    p_train.set_defaults(func=_train)

    # predict
    p_pred = sub.add_parser("predict", help="Predict for a date range.")
    p_pred.add_argument("--start", help="Start date (YYYY-MM-DD).")
    p_pred.add_argument("--end", help="End date (YYYY-MM-DD).")
    p_pred.add_argument(
        "--output", default="predict_output.csv", help="Output CSV path."
    )
    p_pred.set_defaults(func=_predict)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Full walk-forward evaluation.")
    p_eval.add_argument(
        "--start", default="2002-01-01", help="Start date (YYYY-MM-DD)."
    )
    p_eval.add_argument(
        "--end", default="2025-01-01", help="End date (YYYY-MM-DD)."
    )
    p_eval.add_argument(
        "--test-size", type=float, default=0.10, help="Test fraction."
    )
    p_eval.add_argument(
        "--output",
        default="reports/metrics.json",
        help="Output JSON path.",
    )
    p_eval.set_defaults(func=_evaluate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())