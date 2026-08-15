"""
Market data service — online / offline abstraction layer for Yahoo Finance.

Provides a single :class:`MarketDataService` interface that can be backed
by either **live yfinance API calls** or **pre-downloaded serialised files**
stored under ``<data_dir>/<SYMBOL>/`` (as created by
``scripts/yahoo_finance_download.py``).

Usage
-----
::

    from fin_ai.core.service import MarketDataService

    service = MarketDataService.from_environment()
    info = service.get_stock_info("AAPL")
    df = service.get_stock_data("AAPL", "2024-01-01", "2024-12-31")
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from fin_ai.config import YAHOO_SERVICE_OFFLINE, YAHOO_DATA_DIR

# ---------------------------------------------------------------------------
# Import sibling exceptions (late-import friendly, but safe at top level)
# ---------------------------------------------------------------------------

from fin_ai.core.exceptions import (
    MarketDataNotFoundError,
    MarketDataServiceError,
)

# ---------------------------------------------------------------------------
# Internal helpers shared by both implementations
# ---------------------------------------------------------------------------

_DATA_TYPE_FILENAMES: dict[str, str] = {
    "stock_data": "stock_data",
    "stock_info": "stock_info",
    "company_info": "company_info",
    "dividends": "dividends",
    "income_stmt": "income_stmt",
    "balance_sheet": "balance_sheet",
    "cash_flow": "cash_flow",
    "analyst_recommendations": "analyst_recommendations",
}

_SERIALISATION_EXTENSIONS = [".parquet", ".pkl", ".csv", ".json"]


def _path_for(
    data_dir: Path,
    symbol: str,
    data_type: str,
) -> Path | None:
    """Return the path to the first existing file for *data_type* under
    ``data_dir / <SYMBOL>``, checking all supported extensions."""
    ticker_dir = data_dir / symbol.upper()
    if not ticker_dir.is_dir():
        return None
    stem = _DATA_TYPE_FILENAMES.get(data_type, data_type)
    for ext in _SERIALISATION_EXTENSIONS:
        candidate = ticker_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _load_dataframe(filepath: Path) -> pd.DataFrame:
    """Load a DataFrame from a file in any supported format."""
    ext = filepath.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(filepath)
    elif ext == ".pkl":
        obj = pd.read_pickle(filepath)
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, pd.Series):
            return obj.to_frame(name=obj.name or "value")
        return pd.DataFrame(obj)
    elif ext == ".csv":
        return pd.read_csv(filepath, index_col=0)
    elif ext == ".json":
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list) and data:
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.DataFrame([data])
        return pd.DataFrame()
    raise MarketDataServiceError(
        f"Unsupported file extension: {ext}",
        data_type=filepath.stem,
    )


def _load_dict(filepath: Path) -> dict:
    """Load a dict from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _to_json_value(value: Any) -> Any:
    """Convert pandas/numpy types to JSON-safe Python types."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _to_json_key(value: Any) -> str | int | float | bool | None:
    normalized = _to_json_value(value)
    if isinstance(normalized, (str, int, float, bool)) or normalized is None:
        return normalized
    return str(normalized)


def _dataframe_to_records(frame: pd.DataFrame, max_rows: int = 200) -> dict:
    """Convert a DataFrame to the standard ``{"row_count", "truncated", "records"}`` format."""
    if frame is None or frame.empty:
        return {"row_count": 0, "truncated": False, "records": []}

    converted = frame.copy()
    if isinstance(converted.index, pd.DatetimeIndex):
        converted.index = converted.index.strftime("%Y-%m-%d")

    records = converted.reset_index().to_dict(orient="records")
    json_records = [
        {_to_json_key(k): _to_json_value(v) for k, v in row.items()}
        for row in records[:max_rows]
    ]
    return {
        "row_count": len(records),
        "truncated": len(records) > max_rows,
        "records": json_records,
    }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class MarketDataService(ABC):
    """Abstract interface for market data access (Yahoo Finance).

    Callers should obtain an instance via :meth:`from_environment` rather
    than instantiating a concrete subclass directly.
    """

    @classmethod
    def from_environment(cls) -> MarketDataService:
        """Factory: return the appropriate implementation based on config.

        * If :data:`YAHOO_SERVICE_OFFLINE` is ``True``, returns an
          :class:`OfflineMarketDataService` that reads from serialised files.
        * Otherwise returns an :class:`OnlineMarketDataService` that queries
          the live yfinance API.

        If offline mode is enabled but the data directory does not exist
        (or is empty), a warning message is printed and the **online**
        service is returned as a fallback.
        """
        if not YAHOO_SERVICE_OFFLINE:
            return OnlineMarketDataService()

        data_dir = Path(YAHOO_DATA_DIR)
        if not data_dir.is_dir() or not any(data_dir.iterdir()):
            print(
                f"[fin_ai] YAHOO_SERVICE_OFFLINE=True but data directory "
                f"'{YAHOO_DATA_DIR}' does not exist or is empty. "
                f"Falling back to online service."
            )
            return OnlineMarketDataService()

        return OfflineMarketDataService(data_dir)

    # -- Data accessors (mirror tools.py signatures) -------------------

    @abstractmethod
    def get_stock_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Return historical OHLCV price data as a DataFrame."""

    @abstractmethod
    def get_stock_info(self, symbol: str) -> dict:
        """Return stock metadata / quote info dict."""

    @abstractmethod
    def get_company_info(self, symbol: str) -> pd.DataFrame:
        """Return a single-row DataFrame with company profile fields."""

    @abstractmethod
    def get_stock_dividends(self, symbol: str) -> pd.DataFrame:
        """Return dividend history as a DataFrame."""

    @abstractmethod
    def get_income_stmt(self, symbol: str) -> pd.DataFrame:
        """Return the most recent income statement as a DataFrame."""

    @abstractmethod
    def get_balance_sheet(self, symbol: str) -> pd.DataFrame:
        """Return the most recent balance sheet as a DataFrame."""

    @abstractmethod
    def get_cash_flow(self, symbol: str) -> pd.DataFrame:
        """Return the most recent cash-flow statement as a DataFrame."""

    @abstractmethod
    def get_analyst_recommendations(self, symbol: str) -> pd.DataFrame:
        """Return the analyst recommendations as a DataFrame."""


# ---------------------------------------------------------------------------
# Online (live yfinance API) implementation
# ---------------------------------------------------------------------------


class OnlineMarketDataService(MarketDataService):
    """Service backed by live yfinance API calls."""

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        return yf.Ticker(symbol)

    def get_stock_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        try:
            df = self._get_ticker(symbol).history(start=start_date, end=end_date)
            if df.empty:
                raise MarketDataNotFoundError(
                    symbol, "stock_data",
                    detail="yfinance returned an empty DataFrame.",
                )
            return df
        except MarketDataNotFoundError:
            raise
        except Exception as exc:
            raise MarketDataServiceError(
                f"Failed to fetch stock_data for '{symbol}': {exc}",
                symbol=symbol, data_type="stock_data", cause=exc,
            ) from exc

    def get_stock_info(self, symbol: str) -> dict:
        try:
            info = self._get_ticker(symbol).info
            if not info:
                raise MarketDataNotFoundError(
                    symbol, "stock_info",
                    detail="yfinance returned empty info.",
                )
            return {k: _to_json_value(v) for k, v in info.items()}
        except MarketDataNotFoundError:
            raise
        except Exception as exc:
            raise MarketDataServiceError(
                f"Failed to fetch stock_info for '{symbol}': {exc}",
                symbol=symbol, data_type="stock_info", cause=exc,
            ) from exc

    def get_company_info(self, symbol: str) -> pd.DataFrame:
        try:
            info = self._get_ticker(symbol).info
            return pd.DataFrame([{
                "Company Name": info.get("shortName", "N/A"),
                "Industry": info.get("industry", "N/A"),
                "Sector": info.get("sector", "N/A"),
                "Country": info.get("country", "N/A"),
                "Website": info.get("website", "N/A"),
            }])
        except Exception as exc:
            raise MarketDataServiceError(
                f"Failed to fetch company_info for '{symbol}': {exc}",
                symbol=symbol, data_type="company_info", cause=exc,
            ) from exc

    def get_stock_dividends(self, symbol: str) -> pd.DataFrame:
        try:
            div = self._get_ticker(symbol).dividends
            if isinstance(div, pd.Series):
                df = div.to_frame(name="dividend")
            else:
                df = div
            if df.empty:
                raise MarketDataNotFoundError(
                    symbol, "dividends",
                    detail="yfinance returned empty dividend data.",
                )
            return df
        except MarketDataNotFoundError:
            raise
        except Exception as exc:
            raise MarketDataServiceError(
                f"Failed to fetch dividends for '{symbol}': {exc}",
                symbol=symbol, data_type="dividends", cause=exc,
            ) from exc

    def get_income_stmt(self, symbol: str) -> pd.DataFrame:
        try:
            df = self._get_ticker(symbol).financials
            if df.empty:
                raise MarketDataNotFoundError(
                    symbol, "income_stmt",
                    detail="yfinance returned an empty income statement.",
                )
            return df
        except MarketDataNotFoundError:
            raise
        except Exception as exc:
            raise MarketDataServiceError(
                f"Failed to fetch income_stmt for '{symbol}': {exc}",
                symbol=symbol, data_type="income_stmt", cause=exc,
            ) from exc

    def get_balance_sheet(self, symbol: str) -> pd.DataFrame:
        try:
            df = self._get_ticker(symbol).balance_sheet
            if df.empty:
                raise MarketDataNotFoundError(
                    symbol, "balance_sheet",
                    detail="yfinance returned an empty balance sheet.",
                )
            return df
        except MarketDataNotFoundError:
            raise
        except Exception as exc:
            raise MarketDataServiceError(
                f"Failed to fetch balance_sheet for '{symbol}': {exc}",
                symbol=symbol, data_type="balance_sheet", cause=exc,
            ) from exc

    def get_cash_flow(self, symbol: str) -> pd.DataFrame:
        try:
            df = self._get_ticker(symbol).cashflow
            if df.empty:
                raise MarketDataNotFoundError(
                    symbol, "cash_flow",
                    detail="yfinance returned an empty cash flow statement.",
                )
            return df
        except MarketDataNotFoundError:
            raise
        except Exception as exc:
            raise MarketDataServiceError(
                f"Failed to fetch cash_flow for '{symbol}': {exc}",
                symbol=symbol, data_type="cash_flow", cause=exc,
            ) from exc

    def get_analyst_recommendations(self, symbol: str) -> pd.DataFrame:
        try:
            recs = self._get_ticker(symbol).recommendations
            if recs.empty:
                raise MarketDataNotFoundError(
                    symbol, "analyst_recommendations",
                    detail="yfinance returned empty recommendations.",
                )
            return recs
        except MarketDataNotFoundError:
            raise
        except Exception as exc:
            raise MarketDataServiceError(
                f"Failed to fetch analyst_recommendations for '{symbol}': {exc}",
                symbol=symbol, data_type="analyst_recommendations", cause=exc,
            ) from exc


# ---------------------------------------------------------------------------
# Offline (serialised-file) implementation
# ---------------------------------------------------------------------------


class OfflineMarketDataService(MarketDataService):
    """Service backed by pre-downloaded serialised files.

    Files are expected under ``<data_dir>/<SYMBOL>/<data_type>.<ext>``
    as created by ``scripts/yahoo_finance_download.py``.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)

    # -- internal helpers ----------------------------------------------

    def _resolve(self, symbol: str, data_type: str) -> Path:
        """Resolve the file path or raise :class:`MarketDataNotFoundError`."""
        path = _path_for(self._data_dir, symbol, data_type)
        if path is None:
            raise MarketDataNotFoundError(
                symbol, data_type,
                detail=(
                    f"No serialised file found in "
                    f"'{self._data_dir / symbol.upper()}/'."
                ),
            )
        return path

    def _load_df(self, symbol: str, data_type: str) -> pd.DataFrame:
        try:
            path = self._resolve(symbol, data_type)
            return _load_dataframe(path)
        except MarketDataNotFoundError:
            raise
        except Exception as exc:
            raise MarketDataServiceError(
                f"Failed to load {data_type} for '{symbol}' from offline "
                f"storage: {exc}",
                symbol=symbol, data_type=data_type, cause=exc,
            ) from exc

    # -- public accessors ----------------------------------------------

    def get_stock_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        df = self._load_df(symbol, "stock_data")
        # Filter by date range if the DataFrame has a DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.loc[start_date:end_date]
        return df

    def get_stock_info(self, symbol: str) -> dict:
        path = self._resolve(symbol, "stock_info")
        # The download script saves stock_info as parquet (DataFrame-wrapped dict)
        # or JSON.  Try dict-first, then DataFrame-to-dict.
        if path.suffix == ".json":
            return _load_dict(path)
        df = _load_dataframe(path)
        if not df.empty:
            return df.iloc[0].to_dict()
        return {}

    def get_company_info(self, symbol: str) -> pd.DataFrame:
        return self._load_df(symbol, "company_info")

    def get_stock_dividends(self, symbol: str) -> pd.DataFrame:
        return self._load_df(symbol, "dividends")

    def get_income_stmt(self, symbol: str) -> pd.DataFrame:
        return self._load_df(symbol, "income_stmt")

    def get_balance_sheet(self, symbol: str) -> pd.DataFrame:
        return self._load_df(symbol, "balance_sheet")

    def get_cash_flow(self, symbol: str) -> pd.DataFrame:
        return self._load_df(symbol, "cash_flow")

    def get_analyst_recommendations(self, symbol: str) -> pd.DataFrame:
        return self._load_df(symbol, "analyst_recommendations")
