"""Evaluation module with walk-forward validation and baseline comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


def compute_metrics(y_true, y_pred):
    r2 = float(r2_score(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "r2": round(r2, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 2),
        "mae": round(mae, 3),
    }


def naive_baseline_predictions(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Naive baseline: predict yesterday's (t-1) close as today's (t) close."""
    return X["BAC_lag1"]


def walk_forward_evaluate(X, y, fitted_models, n_splits=5, test_size=0.10):
    """Evaluate models using time-series walk-forward cross-validation."""
    results = {}
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for model_name, model in fitted_models.items():
        cv_r2 = []
        cv_rmse = []

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_model = clone(model)
            fold_model.fit(X_tr, y_tr)
            y_pred = fold_model.predict(X_val)

            cv_r2.append(r2_score(y_val, y_pred))
            cv_rmse.append(np.sqrt(mean_squared_error(y_val, y_pred)))

        results[model_name] = {
            "cv_r2_mean": round(float(np.mean(cv_r2)), 4),
            "cv_r2_std": round(float(np.std(cv_r2)), 4),
            "cv_rmse_mean": round(float(np.mean(cv_rmse)), 2),
            "cv_rmse_std": round(float(np.std(cv_rmse)), 2),
        }

    split_idx = int(len(X) * (1 - test_size))
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    for model_name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        results[model_name]["test"] = compute_metrics(y_test.values, y_pred)

    naive_pred = naive_baseline_predictions(X_test, y_test)
    results["NaiveBaseline"] = {
        "test": compute_metrics(y_test.values, naive_pred.values),
    }

    return results


def print_report(results):
    print("\n" + "=" * 70)
    print("WALK-FORWARD EVALUATION REPORT")
    print("=" * 70)

    for model_name, scores in results.items():
        print(f"\n[{model_name}]")
        if "cv_r2_mean" in scores:
            print(f"  Walk-Forward CV R2  : {scores['cv_r2_mean']:.4f} ± {scores['cv_r2_std']:.4f}")
            print(f"  Walk-Forward CV RMSE : {scores['cv_rmse_mean']:.2f} ± {scores['cv_rmse_std']:.2f}")
        if "test" in scores:
            print(f"  Hold-out Test R2     : {scores['test']['r2']:.4f}")
            print(f"  Hold-out Test RMSE   : {scores['test']['rmse']:.2f}")
            print(f"  Hold-out Test MAE    : {scores['test']['mae']:.3f}")


def save_results(results, path):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")
