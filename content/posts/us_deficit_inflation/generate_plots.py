#!/usr/bin/env python3
"""
Script to generate interactive HTML plots for the US deficit inflation analysis.
This script extracts the key plotting functionality and generates the HTML files
that will be embedded in the Hugo markdown.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

# Set output directory
PLOT_OUTPUT_DIR = "."
plot_counter = 0

@dataclass
class FredMetric:
    name: str
    description: str
    units: str
    frequency: str
    data: Optional[pd.DataFrame] = field(default=None, repr=False)

def plot_dual_axis_series(metrics_container: Dict[str, FredMetric], y1_keys: List[str], y2_keys: List[str], save_html: bool = True, plot_name: str = None):
    """
    Plot multiple time series on a dual y-axis plot using Plotly.
    """
    global plot_counter

    fig = go.Figure()

    # Plot series on y1 axis
    for key in y1_keys:
        data = metrics_container[key].data
        fig.add_trace(go.Scatter(
            x=data["timestamp"],
            y=data["value"],
            mode="lines+markers",
            name=key,
            yaxis="y1"
        ))

    # Plot series on y2 axis
    for key in y2_keys:
        data = metrics_container[key].data
        fig.add_trace(go.Scatter(
            x=data["timestamp"],
            y=data["value"],
            mode="lines+markers",
            name=key,
            yaxis="y2"
        ))

    # Use the first key in each group for axis labels
    y1_label = metrics_container[y1_keys[0]].units if y1_keys else ""
    y2_label = metrics_container[y2_keys[0]].units if y2_keys else ""

    fig.update_layout(
        width=1200,
        height=600,
        xaxis=dict(
            title="Date",
            minor=dict(ticks="outside"),
            dtick="M120",
            tickformat="%Y"
        ),
        yaxis=dict(
            title=y1_label,
            side="left"
        ),
        yaxis2=dict(
            title=y2_label,
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(x=0.00, y=1.16, bgcolor="rgba(255,255,255,0.5)")
    )
    
    # Save as HTML if requested
    if save_html:
        if plot_name is None:
            plot_counter += 1
            plot_name = f"plot_{plot_counter}"
        
        html_file = os.path.join(PLOT_OUTPUT_DIR, f"{plot_name}.html")
        fig.write_html(
            html_file,
            config={'displayModeBar': True, 'responsive': True},
            include_plotlyjs='cdn'
        )
        print(f"💾 Saved interactive plot to: {html_file}")
        print(f"📝 In markdown, use: {{{{< plotly file=\"{plot_name}.html\" height=\"650px\" >}}}}")
    
    return fig

def calculate_and_plot_cross_correlation(
    metrics_container: dict, 
    key1: str, 
    key2: str, 
    max_lag: int = 10,
    save_html: bool = True,
    plot_name: str = None
) -> tuple:
    """
    Calculates and plots the Pearson product-moment correlation coefficients.
    """
    global plot_counter
    
    y1 = metrics_container[key1].data.value.values
    y2 = metrics_container[key2].data.value.values

    # Sanity checks
    if len(y1) != len(y2):
        print(f"Error: Series lengths do not match for {key1} and {key2}.")
        return None, None
    N = len(y1)
    if N < 2:
        print("Error: Less than 2 data points. Cannot calculate correlation.")
        return None, None
    
    # Create lag array and initialize lists for CCF values
    lags_array = np.arange(-max_lag, max_lag + 1)
    ccf_values = []

    # Calculate cross-correlation for each lag
    for lag_val in lags_array:
        abs_lag = abs(lag_val)
        current_overlap_length = N - abs_lag

        if current_overlap_length < 2:
            ccf_values.append(np.nan)
            continue

        if lag_val == 0:
            s1_segment = y1
            s2_segment = y2
        elif lag_val > 0:
            s1_segment = y1[lag_val:]
            s2_segment = y2[:-lag_val]
        else:
            s1_segment = y1[:current_overlap_length]
            s2_segment = y2[abs_lag:]
        
        if len(s1_segment) < 2 or len(s2_segment) < 2:
            ccf_values.append(np.nan)
        elif np.all(s1_segment == s1_segment[0]) or np.all(s2_segment == s2_segment[0]):
            ccf_values.append(np.nan)
        else:
            ccf_values.append(np.corrcoef(s1_segment, s2_segment)[0, 1])
    
    fig_ccf = go.Figure()
    fig_ccf.add_trace(go.Scatter(
        x=lags_array,
        y=ccf_values,
        mode="lines+markers",
        name="CCF"
    ))
    fig_ccf.update_layout(
        title=f"Cross-Correlation: {key1} vs {key2}",
        xaxis_title="Lag (Time Periods)",
        yaxis_title="Cross-Correlation Coefficient",
        width=900,
        height=450,
        showlegend=False,
        xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='Black', minor=dict(ticks="outside")),
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey', range=[-1,1], minor=dict(ticks="outside"))
    )
    fig_ccf.add_shape(type="line",
        x0=lags_array[0] if len(lags_array)>0 else 0, 
        y0=0, 
        x1=lags_array[-1] if len(lags_array)>0 else 0, 
        y1=0,
        line=dict(color="Black", width=1, dash="dash")
    )
    
    # Save as HTML if requested
    if save_html:
        if plot_name is None:
            plot_counter += 1
            plot_name = f"plot_{plot_counter}"
        
        html_file = os.path.join(PLOT_OUTPUT_DIR, f"{plot_name}.html")
        fig_ccf.write_html(
            html_file,
            config={'displayModeBar': True, 'responsive': True},
            include_plotlyjs='cdn'
        )
        print(f"💾 Saved interactive plot to: {html_file}")
        print(f"📝 In markdown, use: {{{{< plotly file=\"{plot_name}.html\" >}}}}")
    
    return lags_array, ccf_values

def main():
    """
    Main function to load data and generate plots.
    Note: You need to run the full notebook first to generate the processed data,
    or this script will download and process the raw FRED data.
    """
    print("Interactive Plot Generator for US Deficit vs Inflation Analysis")
    print("=" * 70)
    print("\nTo generate all interactive plots:")
    print("1. Run all cells in the Jupyter notebook (index.ipynb)")
    print("2. The notebook will automatically generate HTML files for each plot")
    print("\nAlternatively, you can:")
    print("- Open the notebook in Jupyter")
    print("- Run: Kernel -> Restart & Run All")
    print("\nThe following HTML files will be created:")
    print("  - deficit_vs_price_wage_quarterly.html")
    print("  - deficit_vs_inflation_annually.html")
    print("  - ccf_deficit_inflation_levels.html")
    print("  - ccf_deficit_inflation_differenced.html")
    print("  - inflation_vs_log_inflation.html")
    print("\nThese files should then be referenced in your markdown file.")
    print("=" * 70)

if __name__ == "__main__":
    main()
