#!/usr/bin/env python3
"""Build a self-contained Plotly HTML chart showing the past 20 years of:

    - S&P 500 Total Return (Yahoo Finance ^SP500TR)
    - Case-Shiller home price indices for SF, NY, LA, Boston (FRED)

All series are aligned to the latest common start month, normalised to 100
at that point, and the legend reports the annualised total return (CAGR)
over the displayed window.

Output: static/posts/rent-vs-buy/historical_reference.html

Dependencies: pandas, plotly, yfinance. Run from the repo root with any Python
that has them installed, e.g.:

    python scripts/historical_reference.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

# 20-year window ending today, anchored to a calendar month for a clean axis.
END_MONTH = pd.Timestamp.today().to_period("M").to_timestamp()
START_MONTH = (END_MONTH - pd.DateOffset(years=20)).to_period("M").to_timestamp()

CASE_SHILLER_SERIES = {
    "San Francisco": "SFXRSA",
    "New York": "NYXRSA",
    "Los Angeles": "LXXRSA",
    "Boston": "BOXRSA",
}


def fetch_fred(series_id: str) -> pd.Series:
    """Download a FRED CSV and return it as a monthly series.

    FRED encodes missing observations as '.' which pandas reads as object dtype.
    Coerce to numeric and drop NaNs so downstream normalisation/CAGR math gets
    a clean float series.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url, parse_dates=["observation_date"])
    s = pd.to_numeric(df.set_index("observation_date")[series_id], errors="coerce")
    return s.dropna()


def fetch_sp500_tr() -> pd.Series:
    """Monthly S&P 500 Total Return from Yahoo Finance.

    Returns a single series of month-start-dated closes to match FRED.

    yfinance's column layout depends on version and parameters: for a single
    ticker it sometimes returns a flat `(field)` index and sometimes a
    `(field, ticker)` MultiIndex. Handle both rather than depending on a
    specific shape.
    """
    df = yf.download(
        "^SP500TR",
        start=START_MONTH,
        interval="1mo",
        progress=False,
        auto_adjust=False,
    )
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        # MultiIndex case: columns are tickers under the "Close" field.
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    # Yahoo stamps each monthly row with the month-start date already; force it
    # explicitly so the index lines up with FRED's month-start convention.
    close.index = close.index.to_period("M").to_timestamp()
    return close


def cagr(series: pd.Series) -> float:
    """Annualised compound growth between the first and last observations."""
    span_years = (series.index[-1] - series.index[0]).days / 365.25
    return (series.iloc[-1] / series.iloc[0]) ** (1 / span_years) - 1


def build_chart() -> str:
    """Assemble all series, normalise, and return a self-contained HTML string."""
    raw: dict[str, pd.Series] = {"S&P 500 (total return)": fetch_sp500_tr()}
    for label, sid in CASE_SHILLER_SERIES.items():
        raw[label] = fetch_fred(sid)

    # Clip each series to the 20-year window, then align to the latest common
    # start month so every line begins at the same point.
    clipped = {k: v.loc[(v.index >= START_MONTH) & (v.index <= END_MONTH)] for k, v in raw.items()}
    common_start = max(s.index[0] for s in clipped.values())
    common_end = min(s.index[-1] for s in clipped.values())
    aligned = {k: v.loc[(v.index >= common_start) & (v.index <= common_end)] for k, v in clipped.items()}

    # Normalise to 100 at common_start so all lines share a starting point.
    normalised = {k: 100.0 * v / v.iloc[0] for k, v in aligned.items()}

    # Highlight S&P 500 with a warmer hue; cities use a cool palette so the
    # eye reads "stock vs houses" rather than "five competing series".
    colors = {
        "S&P 500 (total return)": "#c43820",
        "San Francisco": "#1f6f8b",
        "New York": "#2e9cca",
        "Los Angeles": "#5b9bd5",
        "Boston": "#0b3954",
    }
    widths = {"S&P 500 (total return)": 3.0}

    fig = go.Figure()
    for label, series in normalised.items():
        annual = cagr(series)
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=f"{label} — {annual * 100:.1f}% CAGR",
                line=dict(color=colors[label], width=widths.get(label, 2.0)),
                hovertemplate="<b>%{fullData.name}</b><br>%{x|%b %Y}<br>Index = %{y:.0f}<extra></extra>",
            )
        )

    window_label = f"{common_start:%b %Y} – {common_end:%b %Y}"
    fig.update_layout(
        title=dict(
            text=f"S&P 500 vs Case-Shiller home prices, {window_label} (indexed to 100)",
            x=0.5,
            xanchor="center",
            font=dict(size=15),
        ),
        yaxis=dict(
            type="log",
            title="Index (log scale, start = 100)",
            gridcolor="#e5e5e5",
        ),
        xaxis=dict(title=None, gridcolor="#e5e5e5"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(l=60, r=20, t=60, b=80),
        paper_bgcolor="#fafafa",
        plot_bgcolor="#ffffff",
        font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif', size=12),
        hovermode="x unified",
    )

    return fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": False, "responsive": True},
    )


def main() -> None:
    out_path = Path(__file__).resolve().parent.parent / "static" / "posts" / "rent-vs-buy" / "historical_reference.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = build_chart()
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
