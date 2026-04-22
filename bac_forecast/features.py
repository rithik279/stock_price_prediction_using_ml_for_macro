"""Feature engineering module for BAC next-day price prediction."""

import pandas as pd


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Engineer lagged and technical features from raw price data.

    All equity and macro features are shifted by 1 trading day (t-1) to
    prevent look-ahead bias. Technical indicators are computed on BAC and
    shifted by 1 as well.

    Args:
        raw: DataFrame with columns BAC, JPM, MS, C, WFC, SPY, ^VIX,
             ^TNX, DX-Y.NYB, CL=F, GC=F indexed by date.

    Returns:
        DataFrame with engineered features and Target column (BAC close at t).
    """
    df = pd.DataFrame(index=raw.index)

    # --- Lagged equity features (t-1) ---
    df["BAC_lag1"] = raw["BAC"].shift(1)
    df["JPM_lag1"] = raw["JPM"].shift(1)
    df["MS_lag1"] = raw["MS"].shift(1)
    df["C_lag1"] = raw["C"].shift(1)
    df["WFC_lag1"] = raw["WFC"].shift(1)
    df["SPY_lag1"] = raw["SPY"].shift(1)

    # --- Lagged macro features (t-1) ---
    df["VIX_lag1"] = raw["^VIX"].shift(1)
    df["Yield10Y_lag1"] = raw["^TNX"].shift(1)
    df["Gold_lag1"] = raw["GC=F"].shift(1)
    df["DXY_lag1"] = raw["DX-Y.NYB"].shift(1)
    df["CrudeOil_lag1"] = raw["CL=F"].shift(1)

    # --- Technical indicators (lagged) ---
    df["BAC_MA5"] = raw["BAC"].rolling(window=5).mean().shift(1)
    df["BAC_MA10"] = raw["BAC"].rolling(window=10).mean().shift(1)
    df["BAC_Vol5"] = raw["BAC"].pct_change(5).shift(1)

    # --- Target: BAC close at t (next-day relative to features) ---
    df["Target"] = raw["BAC"].shift(-1)

    df = df.dropna()
    return df
