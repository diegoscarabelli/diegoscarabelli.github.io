# Interactive Plotly Plots in Hugo + GitHub Pages

## Implementation Summary

This solution enables fully interactive Plotly plots on your Hugo-based GitHub Pages site by exporting plots as HTML files and embedding them using a custom shortcode.

## ✅ What's Been Implemented

### 1. Hugo Shortcode

Created `/layouts/shortcodes/plotly.html` that embeds HTML plot files:

```html
{{- $file := .Get "file" -}}
{{- $height := .Get "height" | default "500px" -}}

<div style="width: 100%; height: {{ $height }};">
  <iframe 
    src="{{ $file }}" 
    style="width: 100%; height: 100%; border: none;"
    frameborder="0"
    scrolling="no">
  </iframe>
</div>
```

### 2. Updated Notebook Functions

Modified the plotting functions in `index.ipynb`:
- Added `os` import for file operations
- Added global `plot_counter` variable
- Updated `plot_dual_axis_series()` with `save_html` and `plot_name` parameters
- Updated `calculate_and_plot_cross_correlation()` with `save_html` and `plot_name` parameters
- All plot calls now use descriptive names:
  - `deficit_vs_price_wage_quarterly`
  - `deficit_vs_inflation_annually`
  - `ccf_deficit_inflation_levels`
  - `ccf_deficit_inflation_differenced`
  - `inflation_vs_log_inflation`

## 📋 Next Steps

### Step 1: Generate HTML Plot Files

1. Open `index.ipynb` in Jupyter Notebook or VS Code
2. Run all cells (Kernel → Restart & Run All)
3. The notebook will automatically create 5 HTML files in the same directory

When each plot is created, you'll see output like:
```
💾 Saved interactive plot to: deficit_vs_price_wage_quarterly.html
📝 In markdown, use: {{< plotly file="deficit_vs_price_wage_quarterly.html" height="650px" >}}
```

### Step 2: Convert Notebook to Markdown

Run the standard conversion:
```bash
jupyter nbconvert --to markdown index.ipynb
```

This will create/update `index.md` with static images.

### Step 3: Update Markdown with Interactive Plots

Edit `index.md` and replace the static image references with the shortcode.

**Find and replace:**

1. **After "Visual Inspection" section** - Replace the first plot image with:
   ```markdown
   {{< plotly file="deficit_vs_price_wage_quarterly.html" height="650px" >}}
   ```

2. **After "Calculate Annual Percentage Change" section** - Replace the second plot with:
   ```markdown
   {{< plotly file="deficit_vs_inflation_annually.html" height="650px" >}}
   ```

3. **In "Cross-Correlation of Undifferenced (Level) Series"** - Replace the CCF plot with:
   ```markdown
   {{< plotly file="ccf_deficit_inflation_levels.html" height="500px" >}}
   ```

4. **In "Cross-Correlation of Differenced Series"** - Replace the differenced CCF plot with:
   ```markdown
   {{< plotly file="ccf_deficit_inflation_differenced.html" height="500px" >}}
   ```

5. **After log transformation comparison** - Replace the log comparison plot with:
   ```markdown
   {{< plotly file="inflation_vs_log_inflation.html" height="650px" >}}
   ```

### Step 4: Test Locally

```bash
cd /Users/diegoscarabelli/repos/diegoscarabelli.github.io
hugo server
```

Visit http://localhost:1313 and navigate to your post to verify the interactive plots work.

### Step 5: Commit and Deploy

```bash
git add .
git commit -m "Add interactive Plotly plots using HTML export"
git push origin interactive-plots
```

Then create a pull request to merge into main.

## How It Works

### Notebook Side

When you run a cell that creates a plot, the function now:
1. Creates the Plotly figure
2. Saves it as a standalone HTML file using `fig.write_html()`
3. Uses Plotly.js from CDN to keep files small
4. Displays the figure in the notebook (for development)
5. Prints the shortcode syntax for easy copy-paste

### Hugo Side

The shortcode:
1. Takes the HTML filename and optional height as parameters
2. Creates an iframe to embed the HTML file
3. The HTML file contains the full interactive Plotly plot

### Deployment

When you push to GitHub:
1. GitHub Pages serves the static HTML files alongside your markdown
2. Users get fully interactive plots (zoom, pan, hover, etc.)
3. No special build steps or JavaScript frameworks needed

## Benefits

✅ **Fully interactive** - Zoom, pan, hover for exact values  
✅ **No build complexity** - Works with standard Hugo + GitHub Pages  
✅ **Lightweight** - CDN-hosted Plotly.js keeps HTML files small  
✅ **Maintains workflow** - Develop in Jupyter, publish as usual  
✅ **Version controlled** - HTML files tracked in git  

## Troubleshooting

**Plots don't appear on the site:**
- Ensure HTML files are in the same directory as the markdown
- Check that file names match exactly in the shortcode
- Verify files are committed and pushed to GitHub
- Check browser console for loading errors

**Plots are cut off or too small:**
- Adjust the `height` parameter in the shortcode
- Use CSS units: "500px", "80vh", etc.

**Want plots to work offline:**
- In the notebook, change `include_plotlyjs='cdn'` to `include_plotlyjs=True`
- This embeds Plotly.js in each HTML file (larger file sizes)

## Future Enhancements

You could create a custom nbconvert template to automatically:
- Replace plot outputs with shortcodes during conversion
- Extract plot names from cell metadata
- Generate the markdown with shortcodes in one step

For now, the manual approach is simple and gives you full control.
