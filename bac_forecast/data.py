"""Data fetching and collection module."""

import pandas as pd
import yfinance as yf


EQUITY_TICKERS = ["BAC", "JPM", "MS", "C", "WFC", "SPY"]
MACRO_TICKERS = ["^VIX", "^TNX", "DX-Y.NYB", "CL=F", "GC=F"]
ALL_TICKERS = EQUITY_TICKERS + MACRO_TICKERS


def fetch_ohlcv(
    tickers: list[str],
    start: str = "2002-01-01",
    end: str = "2025-01-01",
) -> pd.DataFrame:
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
    return data


def fetch_close(
    tickers: list[str],
    start: str = "2002-01-01",
    end: str = "2025-01-01",
) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
    close = raw["Close"]
    close = close.ffill()
    return close


def load_data(
    start: str = "2002-01-01",
    end: str = "2025-01-01",
) -> pd.DataFrame:
    data = fetch_close(ALL_TICKERS, start=start, end=end)
    data = data.ffill()
    return data
