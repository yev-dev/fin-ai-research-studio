import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt

from fin_ai.core import tools

import numpy as np

def _records_to_dataframe(payload: dict) -> pd.DataFrame:
    """Convert a tools payload with 'records' to a pandas DataFrame indexed by date when possible."""
    if not payload or payload.get("row_count", 0) == 0:
        raise ValueError(payload.get("error", "no data returned"))
    records = payload.get("records", [])
    if not records:
        raise ValueError("no records in payload")
    df = pd.DataFrame(records)
    # Common index column names
    for col in ("index", "date", "Date", "datetime"):
        if col in df.columns:
            df = df.rename(columns={col: "date"})
            try:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            except Exception:
                pass
            return df
    return df


def plot_stock_price_chart(
    ticker_symbol: str,
    start_date: str,
    end_date: str,
    save_path: str,
    verbose: bool = False,
    plot_type: str = "line",
    mav: Optional[list[int]] = None,
) -> str:
    """Fetch price data via `tools.get_stock_data` and save a price chart.

    Returns the saved file path or raises on error.
    """
    payload = tools.get_stock_data(ticker_symbol, start_date, end_date)
    if "error" in payload:
        raise RuntimeError(payload["error"])
    df = _records_to_dataframe(payload)
    if verbose:
        print(df.head())

    # Prefer Close series
    if "Close" in df.columns:
        series = df["Close"].astype(float)
    elif "close" in df.columns:
        series = df["close"].astype(float)
    else:
        # Try to find any numeric column
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            raise RuntimeError("no numeric price column found in data")
        series = df[numeric_cols[0]].astype(float)

    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values, label=f"{ticker_symbol} Close")
    if mav:
        for m in mav:
            plt.plot(series.index, series.rolling(window=m).mean(), label=f"MA{m}")
    plt.title(f"{ticker_symbol} price {start_date} → {end_date}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def get_share_performance(ticker_symbol: str, filing_date: str, save_path: str) -> str:
    """Plot % change of a stock vs S&P 500 over the 12 months before `filing_date`.

    Uses `tools.get_stock_data` and `tools.get_stock_info`.
    """
    if isinstance(filing_date, str):
        filing_dt = datetime.strptime(filing_date, "%Y-%m-%d")
    else:
        filing_dt = filing_date
    start = (filing_dt - timedelta(days=365)).strftime("%Y-%m-%d")
    end = filing_dt.strftime("%Y-%m-%d")

    payload_t = tools.get_stock_data(ticker_symbol, start, end)
    payload_sp = tools.get_stock_data("^GSPC", start, end)
    if "error" in payload_t:
        raise RuntimeError(payload_t["error"])
    if "error" in payload_sp:
        raise RuntimeError(payload_sp["error"])

    df_t = _records_to_dataframe(payload_t)
    df_sp = _records_to_dataframe(payload_sp)

    # pick close series
    def close_series(df):
        for c in ("Close", "close"):
            if c in df.columns:
                return df[c].astype(float)
        numeric = df.select_dtypes(include="number").iloc[:, 0]
        return numeric.astype(float)

    t_close = close_series(df_t)
    sp_close = close_series(df_sp)

    company_change = (t_close - t_close.iloc[0]) / t_close.iloc[0] * 100
    sp_change = (sp_close - sp_close.iloc[0]) / sp_close.iloc[0] * 100

    info = tools.get_stock_info(ticker_symbol)
    short_name = info.get("shortName") or ticker_symbol

    plt.figure(figsize=(12, 6))
    plt.plot(company_change.index, company_change.values, label=f"{short_name} Change %")
    plt.plot(sp_change.index, sp_change.values, label="S&P 500 Change %")
    plt.title(f"{short_name} vs S&P 500 - Change % Over The Past Year")
    plt.xlabel("Date")
    plt.ylabel("Change %")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def get_pe_eps_performance(ticker_symbol: str, filing_date: str, years: int = 4, save_path: Optional[str] = None) -> str:
    """Attempt to compute PE over time using `tools.get_income_stmt` and price history.

    This function makes a best-effort extraction of EPS from the income statement
    payload. If EPS cannot be extracted, an informative error is raised.
    """
    if isinstance(filing_date, str):
        filing_dt = datetime.strptime(filing_date, "%Y-%m-%d")
    else:
        filing_dt = filing_date
    days = round((years + 1) * 365.25)
    start = (filing_dt - timedelta(days=days)).strftime("%Y-%m-%d")
    end = filing_dt.strftime("%Y-%m-%d")

    inc_payload = tools.get_income_stmt(ticker_symbol)
    if "error" in inc_payload:
        raise RuntimeError(inc_payload["error"])
    try:
        df_inc = _records_to_dataframe(inc_payload)
    except Exception as exc:
        raise RuntimeError(f"failed to parse income statement: {exc}")

    # Try to find a row that looks like diluted EPS (case-insensitive)
    eps_row = None
    for idx in df_inc.index.astype(str):
        if "eps" in idx.lower() or "diluted" in idx.lower():
            eps_row = df_inc.loc[idx]
            break
    if eps_row is None:
        # Try columns (some income stmts have metrics as columns)
        for col in df_inc.columns:
            if "eps" in str(col).lower() or "diluted" in str(col).lower():
                eps_row = df_inc[col]
                break
    if eps_row is None:
        raise RuntimeError("could not find EPS row/column in income statement payload")

    # eps_row should be a Series with dates or periods in the index; coerce to numeric
    try:
        eps_series = pd.to_numeric(eps_row).dropna()
    except Exception:
        eps_series = eps_row.dropna().astype(float)

    # Align EPS dates to nearest trading date by fetching price history
    price_payload = tools.get_stock_data(ticker_symbol, start, end)
    if "error" in price_payload:
        raise RuntimeError(price_payload["error"])
    df_price = _records_to_dataframe(price_payload)
    if "Close" in df_price.columns:
        price_close = df_price["Close"].astype(float)
    else:
        price_close = df_price.select_dtypes(include="number").iloc[:, 0].astype(float)

    pe_values = []
    pe_dates = []
    # Iterate EPS points and find nearest price
    for period, eps in eps_series.items():
        try:
            period_dt = pd.to_datetime(period)
        except Exception:
            # try parsing from column labels
            period_dt = None
        if period_dt is not None:
            # find nearest index in price_close
            nearest_idx = price_close.index.get_indexer([period_dt], method="nearest")
            if nearest_idx.size and nearest_idx[0] >= 0:
                p = price_close.iloc[nearest_idx[0]]
                pe_values.append(p / float(eps) if float(eps) != 0 else None)
                pe_dates.append(price_close.index[nearest_idx[0]])

    if not pe_values:
        raise RuntimeError("unable to compute PE series from available EPS and price data")

    # Plot PE (left) and EPS (right)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(pe_dates, pe_values, color="tab:blue", label="PE Ratio")
    ax1.set_ylabel("PE Ratio", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(pe_dates, eps_series.iloc[: len(pe_dates)], color="tab:red", label="EPS")
    ax2.set_ylabel("EPS", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title(f"{ticker_symbol} PE Ratios and EPS Over The Past {years} Years")
    fig.tight_layout()
    out_path = save_path or f"{ticker_symbol}_pe_eps.png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_cross_asset_comparison(series_map: dict, start: str, end: str, save_path: str, normalize: bool = True) -> str:
    """Plot multiple series on the same axes for comparison.

    series_map: dict[label] = symbol
    If normalize=True the series are rebased to 100 at start date to compare relative moves.
    """
    frames = {}
    for label, symbol in series_map.items():
        payload = tools.get_stock_data(symbol, start, end)
        if "error" in payload:
            raise RuntimeError(f"{symbol}: {payload['error']}")
        df = _records_to_dataframe(payload)
        # pick Close or first numeric column
        if "Close" in df.columns:
            s = df["Close"].astype(float)
        elif "close" in df.columns:
            s = df["close"].astype(float)
        else:
            s = df.select_dtypes(include="number").iloc[:, 0].astype(float)
        frames[label] = s

    combined = pd.concat(frames, axis=1).sort_index()
    if normalize:
        combined = combined.divide(combined.iloc[0]).multiply(100)

    plt.figure(figsize=(12, 6))
    for col in combined.columns:
        plt.plot(combined.index, combined[col], label=col)
    plt.title(f"Cross-asset comparison {start} → {end}")
    plt.xlabel("Date")
    plt.ylabel("Indexed value" if normalize else "Value")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_time_series(series_map: dict, start: str, end: str, save_path: str, movers: list = None) -> str:
    """Plot one or more time series with optional momentum/indicator overlays.

    series_map: dict[label]=symbol
    movers: list of dicts {"label":..., "window":30} to plot rolling momentum lines
    """
    frames = {}
    for label, symbol in series_map.items():
        payload = tools.get_stock_data(symbol, start, end)
        if "error" in payload:
            raise RuntimeError(f"{symbol}: {payload['error']}")
        df = _records_to_dataframe(payload)
        if "Close" in df.columns:
            s = df["Close"].astype(float)
        else:
            s = df.select_dtypes(include="number").iloc[:, 0].astype(float)
        frames[label] = s

    combined = pd.concat(frames, axis=1).sort_index()
    plt.figure(figsize=(12, 6))
    for col in combined.columns:
        plt.plot(combined.index, combined[col], label=col)

    if movers:
        for m in movers:
            lbl = m.get("label")
            w = m.get("window", 30)
            if lbl in combined.columns:
                plt.plot(combined.index, combined[lbl].rolling(window=w).mean(), linestyle="--", label=f"{lbl} MA{w}")

    plt.title(f"Time series {start} → {end}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_narrative_timeline(events: list, start: str, end: str, save_path: str) -> str:
    """Plot a simple timeline of narrative events.

    events: list of dicts with keys `date` (YYYY-MM-DD) and `label` (short)
    """
    dates = []
    labels = []
    for ev in events:
        try:
            dt = pd.to_datetime(ev.get("date"))
        except Exception:
            continue
        dates.append(dt)
        labels.append(ev.get("label", ""))

    if not dates:
        raise RuntimeError("no valid events provided")

    y = np.arange(len(dates))
    plt.figure(figsize=(12, max(3, len(dates) * 0.3)))
    plt.hlines(1, pd.to_datetime(start), pd.to_datetime(end), colors="lightgray")
    plt.scatter(dates, np.ones_like(dates), s=60, color="C1")
    for i, txt in enumerate(labels):
        plt.annotate(txt, (dates[i], 1), xytext=(0, 10 + (i % 3) * 6), textcoords="offset points", rotation=30)
    plt.gca().get_yaxis().set_visible(False)
    plt.title("Narrative timeline")
    plt.xlabel("Date")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_lead_lag(series_a_symbol: str, series_b_symbol: str, start: str, end: str, save_path: str, max_lag: int = 30) -> str:
    """Compute simple cross-correlation (lagged Pearson) between two series and plot as a bar chart.

    Positive lag means series_a leads series_b by that many days.
    """
    pa = tools.get_stock_data(series_a_symbol, start, end)
    pb = tools.get_stock_data(series_b_symbol, start, end)
    if "error" in pa or "error" in pb:
        raise RuntimeError(f"data error: {pa.get('error')} {pb.get('error')}")
    a = _records_to_dataframe(pa)
    b = _records_to_dataframe(pb)
    # pick Close
    def cs(df):
        if "Close" in df.columns:
            return df["Close"].astype(float)
        return df.select_dtypes(include="number").iloc[:, 0].astype(float)

    sa = cs(a).dropna()
    sb = cs(b).dropna()
    merged = pd.concat([sa, sb], axis=1).dropna()
    sa = merged.iloc[:, 0]
    sb = merged.iloc[:, 1]

    lags = range(-max_lag, max_lag + 1)
    corrs = []
    for lag in lags:
        if lag < 0:
            val = sa.shift(-lag).corr(sb)
        else:
            val = sa.corr(sb.shift(lag))
        corrs.append(val)

    plt.figure(figsize=(10, 5))
    plt.bar(lags, corrs, color="C2")
    plt.axvline(0, color="k", linewidth=0.8)
    plt.title(f"Lead/lag correlation: {series_a_symbol} vs {series_b_symbol}")
    plt.xlabel("Lag (days): positive = first series leads second")
    plt.ylabel("Pearson correlation")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


if __name__ == "__main__":
    # Quick smoke example (will raise if data services are not configured)
    try:
        print(plot_stock_price_chart("AAPL", "2024-03-01", "2024-04-01", "aapl_price.png"))
    except Exception as e:
        print("example run failed:", e)
