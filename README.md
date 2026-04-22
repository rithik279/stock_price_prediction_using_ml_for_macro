# BAC Stock Price Forecasting

**Predicting Bank of America (BAC) next-day closing price using machine learning and macroeconomic indicators.**

A modular, production-ready Python package that replaces the original notebook-first workflow with a structured pipeline: data collection → feature engineering → model training → walk-forward evaluation → CLI + reporting.

---

## Quick Start

```bash
pip install -r requirements.txt

python -m bac_forecast train --start 2002-01-01 --end 2025-01-01
python -m bac_forecast predict --start 2024-11-01 --end 2025-01-01
python -m bac_forecast evaluate --start 2002-01-01 --end 2025-01-01
```

---

## Project Structure

```
bac_forecast/
├── __init__.py
├── __main__.py        # python -m bac_forecast entry point
├── cli.py             # CLI argument parser
├── data.py            # Yahoo Finance data fetching
├── features.py        # Feature engineering (lagged + technical)
├── train.py           # Model training
└── evaluate.py        # Walk-forward CV, baseline comparison, reporting
reports/
└── metrics.json       # Walk-forward evaluation results
notebooks/
└── model_dev.ipynb    # Original exploration notebook
```

---

## Package Overview

### `bac_forecast.data`
Downloads close prices for the full BAC forecast universe from Yahoo Finance:
- **Equities:** BAC, JPM, MS, C, WFC, SPY
- **Macro:** ^VIX (VIX), ^TNX (10Y yield), DX-Y.NYB (USD index), CL=F (crude oil), GC=F (gold)
- Forward-fill applied for non-trading days.

### `bac_forecast.features`
Builds the feature matrix with strict look-ahead bias prevention:
- **Equity features (t-1):** BAC, JPM, MS, C, WFC, SPY closes lagged by 1 day
- **Macro features (t-1):** VIX, 10Y yield, DXY, crude oil, gold lagged by 1 day
- **Technical indicators:** BAC 5-day MA, BAC 10-day MA, 5-day rolling volatility (all lagged)
- **Target:** BAC close at t (effectively next-day price)

### `bac_forecast.train`
Trains and exposes four models:
| Model | Configuration |
|---|---|
| Decision Tree | `max_depth=4` |
| Random Forest | `n_estimators=100` |
| KNN | `n_neighbors=5`, StandardScaler |
| SVR | RBF kernel, StandardScaler |

### `bac_forecast.evaluate`
- **Walk-forward validation** via `TimeSeriesSplit` with `n_splits=5`
- **Naive baseline:** predict yesterday's close (t-1 lag). If no model beats this, it is reported honestly.
- **Hold-out test:** final 10% of the time series
- Results written to `reports/metrics.json`

---

## CLI Commands

### `python -m bac_forecast train`
```bash
python -m bac_forecast train [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                             [--test-size FRAC] [--output CSV]
```
Trains all models, prints evaluation report, saves predictions CSV and metrics JSON.

### `python -m bac_forecast predict`
```bash
python -m bac_forecast predict [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                               [--output CSV]
```
Trains on available history, outputs predictions for the given range.

### `python -m bac_forecast evaluate`
```bash
python -m bac_forecast evaluate [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                                 [--test-size FRAC] [--output JSON]
```
Runs walk-forward CV and hold-out evaluation; saves structured results to JSON.

---

## Features (All t-1 Lagged)

| Category | Features |
|---|---|
| Equities | BAC, JPM, MS, C, WFC, SPY |
| Macro | VIX, 10Y Yield, USD Index, Crude Oil, Gold |
| Technical | BAC_MA5, BAC_MA10, BAC_Vol5 |
| Target | BAC close (t) |

---

## Results Interpretation

The `reports/metrics.json` file contains:
- `cv_r2_mean` / `cv_r2_std`: Walk-forward cross-validation R² across folds
- `cv_rmse_mean` / `cv_rmse_std`: Walk-forward RMSE across folds
- `test.r2` / `test.rmse` / `test.mae`: Hold-out test metrics

If a model's test R² is **below the NaiveBaseline**, that is explicitly reported — this is a learning outcome, not a failure. Beating the naive baseline is non-trivial in stock price prediction.

---

## Limitations

- **Look-ahead bias:** Features are strictly lagged (t-1). No future information leaks.
- **Non-stationarity:** Market regimes shift. Periodic retraining is required.
- **Forward-fill risk:** Stale macro values can be imprinted over long market closures.
- **Overfitting:** Default hyperparameters; use walk-forward CV to tune.
- **Naive baseline:** "Predict yesterday's close" is a strong baseline for daily price series.

---

## Next Steps

- [ ] Add returns, log-returns, z-scored features
- [ ] Add RSI, ATR, rolling skew/kurtosis
- [ ] Pipeline with StandardScaler for all models
- [ ] GridSearchCV with TimeSeriesSplit
- [ ] Add XGBoost / LightGBM
- [ ] Diebold-Mariano test vs baseline
- [ ] Residual analysis, QQ plots, ACF/PACF
- [ ] Rolling/expanding walk-forward backtest
- [ ] Stress-test on 2008 and 2020 windows
- [ ] Streamlit dashboard for on-demand forecasts
- [ ] Model versioning and scheduled retraining

---

## Author

Rithik Singh — Quant Finance Project Series
