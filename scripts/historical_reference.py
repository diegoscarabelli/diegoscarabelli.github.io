#!/usr/bin/env python3
"""Build a Plotly HTML chart showing the past 20 years of:

    - S&P 500 Total Return (Yahoo Finance ^SP500TR)
    - Case-Shiller home price indices for SF, NY, LA, Boston (FRED)

All series are aligned to the latest common start month, normalised to 100
at that point, and the legend reports the annualised total return (CAGR)
over the displayed window.

The generated HTML inlines all chart data but loads plotly.js from the CDN
(`include_plotlyjs="cdn"` below), matching the convention used by the
rent-vs-buy dashboard. The file therefore requires network access at view
time but is byte-cheap to ship.

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

# 20-year window anchored on calendar months for a clean axis. END_MONTH is
# the first day of the current month (today snapped to month-start via Period).
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
    if df is None or df.empty:
        raise RuntimeError(
            "yfinance returned no data for ^SP500TR (network blip, ticker "
            "renamed, or yfinance schema change). Re-run later or inspect "
            "the response manually."
        )
    # `Close` can sit on either the flat columns or the top level of a
    # `(field, ticker)` MultiIndex — check the right level either way.
    top_columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
    if "Close" not in top_columns:
        raise RuntimeError(
            "yfinance response for ^SP500TR has no 'Close' column "
            f"(got {list(top_columns)}). Likely a yfinance schema change."
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
    if span_years <= 0:
        raise ValueError(
            f"Cannot compute CAGR: series spans {span_years:.3f} years "
            f"({series.index[0]:%Y-%m} to {series.index[-1]:%Y-%m})."
        )
    return (series.iloc[-1] / series.iloc[0]) ** (1 / span_years) - 1


def build_chart() -> str:
    """Assemble all series, normalise, and return the chart as an HTML string.

    The returned document inlines all chart data; plotly.js is loaded from
    the CDN at view time (see `include_plotlyjs="cdn"` at the bottom of this
    function).
    """
    raw: dict[str, pd.Series] = {"S&P 500 (total return)": fetch_sp500_tr()}
    for label, sid in CASE_SHILLER_SERIES.items():
        raw[label] = fetch_fred(sid)

    # Clip each series to the 20-year window, then align to the latest common
    # start month so every line begins at the same point.
    clipped = {k: v.loc[(v.index >= START_MONTH) & (v.index <= END_MONTH)] for k, v in raw.items()}
    empty = [k for k, v in clipped.items() if v.empty]
    if empty:
        raise RuntimeError(
            f"No observations in {START_MONTH:%Y-%m} to {END_MONTH:%Y-%m} for: {', '.join(empty)}. "
            "A FRED/yfinance hiccup or a stale ticker is likely; re-run later."
        )
    common_start = max(s.index[0] for s in clipped.values())
    common_end = min(s.index[-1] for s in clipped.values())
    if common_start > common_end:
        raise RuntimeError(
            f"Series do not overlap: latest start {common_start:%Y-%m} > earliest end {common_end:%Y-%m}."
        )
    aligned = {k: v.loc[(v.index >= common_start) & (v.index <= common_end)] for k, v in clipped.items()}

    # Normalise to 1.0 at common_start: y-axis reads directly as a growth
    # multiplier (1x = start, 2x = doubled, 0.5x = halved). Easier to read on a
    # log scale than an index normalised to 100.
    normalised = {k: v / v.iloc[0] for k, v in aligned.items()}

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
                hovertemplate="<b>%{fullData.name}</b><br>%{x|%b %Y}<br>%{y:.2f}× start<extra></extra>",
            )
        )

    window_label = f"{common_start:%b %Y} – {common_end:%b %Y}"
    fig.update_layout(
        # Without an explicit height Plotly's `height:100%` div collapses inside
        # the iframe (no intrinsic body height) and the chart renders near-zero.
        height=600,
        title=dict(
            text=f"S&P 500 vs Case-Shiller home prices, {window_label}",
            x=0.5,
            xanchor="center",
            font=dict(size=15),
        ),
        yaxis=dict(
            type="log",
            title="Growth multiple (log scale)",
            gridcolor="#e5e5e5",
            # Show explicit multiplier ticks (0.5×, 1×, 2×, 5×, 10×) so the
            # axis reads as "how many times the starting value", not as
            # log-decade-with-implicit-multiplier.
            tickvals=[0.5, 1, 2, 3, 5, 7, 10],
            ticktext=["0.5×", "1× (start)", "2×", "3×", "5×", "7×", "10×"],
        ),
        xaxis=dict(title=None, gridcolor="#e5e5e5"),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=60, r=20, t=60, b=80),
        paper_bgcolor="#fafafa",
        plot_bgcolor="#ffffff",
        font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif', size=12),
        hovermode="x unified",
    )

    html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        # Default is "100%" which collapses to zero inside the iframe body
        # (no intrinsic height), so layout.height never gets a chance to apply.
        default_height="600px",
        config={"displayModeBar": True, "responsive": True},
    )
    # Plotly's full_html output ships `<html><head><meta charset="utf-8"/></head>`
    # with no doctype, lang, viewport, or title — which puts the iframed
    # document in quirks mode and weakens a11y metadata. Patch the head once.
    plotly_head = '<html>\n<head><meta charset="utf-8" /></head>'
    patched_head = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>S&amp;P 500 vs Case-Shiller home prices (20-year reference)</title>\n'
        '</head>'
    )
    patched = html.replace(plotly_head, patched_head, 1)
    if patched == html:
        raise RuntimeError(
            "Plotly head-patch failed: anchor string not found. The to_html() "
            "output format likely changed; update `plotly_head` to match."
        )
    return patched


def main() -> None:
    out_path = Path(__file__).resolve().parent.parent / "static" / "posts" / "rent-vs-buy" / "historical_reference.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = build_chart()
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
