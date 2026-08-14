"""Simple runner to demonstrate Morning Analysis pipeline using tools and charting.

This script attempts to fetch real data via `fin_ai.core.tools` and falls back
to synthetic data when the provider is not available. It then uses the
`fin_ai.core.charting` helpers to create PNGs under `./outputs/`.

Usage:
    python scripts/generate_morning_analysis.py
"""

import json
import os
from datetime import datetime, timedelta

from fin_ai.core import tools, charting
import pandas as pd
import numpy as np

OUT_DIR = os.path.abspath("outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Example mapping: theme -> symbol/provider id
THEME_MAP = {
    "Fed policy": "US10Y",
    "Oil": "CL=F",
    "Equities": "SPY",
    "Credit": "CDX.NA.IG",  # provider id; falls back to get_stock_data if needed
    "FX": "DXY",
}

END = datetime.today()
START = END - timedelta(days=180)
START_S = START.strftime("%Y-%m-%d")
END_S = END.strftime("%Y-%m-%d")

charts = []
summary = {"generated_at": datetime.now().isoformat(), "charts": []}

for theme, symbol in THEME_MAP.items():
    try:
        if theme == "Credit":
            payload = tools.get_credit_spread(symbol, START_S, END_S)
        elif theme in ("Fed policy", "FX"):
            payload = tools.get_macro_series(symbol, START_S, END_S)
        else:
            payload = tools.get_stock_data(symbol, START_S, END_S)

        if payload.get("error") or payload.get("row_count", 0) == 0:
            raise RuntimeError(payload.get("error") or "no rows")

        # Prepare series_map with a single series for plotting
        series_map = {theme: symbol}
        save_path = os.path.join(OUT_DIR, f"{theme.replace(' ', '_')}.png")
        # Use plot_time_series for single-series themes
        chart_path = charting.plot_time_series(series_map, START_S, END_S, save_path)
        summary["charts"].append({"theme": theme, "symbol": symbol, "path": chart_path})
        print(f"Wrote chart for {theme}: {chart_path}")
    except Exception as exc:
        # Fallback: create synthetic series and plot
        print(f"Warning: failed to fetch {symbol} ({theme}): {exc}")
        dates = pd.date_range(START, END, freq="B")
        vals = np.cumsum(np.random.randn(len(dates))) + 100
        df = pd.DataFrame({"Date": dates, "Close": vals}).set_index("Date")
        # Save to a temp csv and create payload-like dict so charting helpers can use it
        tmp_csv = os.path.join(OUT_DIR, f"{theme.replace(' ', '_')}_mock.csv")
        df.to_csv(tmp_csv)
        # Use charting directly on DataFrame-like input by saving and plotting via series_map
        series_map = {theme: symbol}
        save_path = os.path.join(OUT_DIR, f"{theme.replace(' ', '_')}_mock.png")
        # Charting expects tools.get_stock_data to return records; instead call plot_time_series
        # using a small wrapper that writes the DataFrame into a CSV then reads via pandas inside charting.
        # Simpler: manually plot here to ensure demonstration works.
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(dates, vals, label=theme)
        plt.title(f"{theme} (mock) {START_S} -> {END_S}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        summary["charts"].append({"theme": theme, "symbol": symbol, "path": save_path, "mock": True})

# Write manifest
manifest_path = os.path.join(OUT_DIR, "morning_analysis_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(summary, f, indent=2)

print("Finished. Manifest:", manifest_path)
