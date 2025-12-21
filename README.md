# No Spoon Labs

## Overview

[**No Spoon Labs**](https://www.nospoonlabs.com) is a personal blog and knowledge site built with [Hugo](https://gohugo.io/) and published via [GitHub Pages](https://pages.github.com/). All content is authored in Markdown and managed in the repository, with Hugo generating the static site. The site is automatically built and deployed using a [GitHub Actions workflow](.github/workflows/hugo.yaml), which installs Hugo, builds the site, and publishes it to GitHub Pages.

### About Hugo

[Hugo](https://gohugo.io/) is a fast, open-source static site generator. It takes Markdown files and templates, processes them, and outputs a complete static website. Hugo is widely used for blogs, documentation, and personal sites because it is easy to use, supports themes, and builds sites extremely quickly. In this project, all posts and pages are written in Markdown under [`content/`](content/), and layouts are defined in [`layouts/`](layouts/). Hugo processes these and outputs the static HTML site to [`public/`](public/).

### About GitHub Pages and GitHub Actions

[GitHub Pages](https://pages.github.com/) is a free static site hosting service that serves content directly from a GitHub repository. When integrated with Hugo, GitHub Pages hosts the generated static files (HTML, CSS, JS, etc.) from the `public/` directory. The [GitHub Actions workflow](.github/workflows/hugo.yaml) in this repository automates the process: on every push to the `main` branch, it builds the site with Hugo and deploys the result to GitHub Pages, making updates live automatically. This workflow:
- Runs on every push to the `main` branch.
- Installs the specified Hugo version.
- Builds the site in production mode.
- Deploys the generated static files to GitHub Pages.

### Analytics with GoatCounter

Page view analytics are powered by [GoatCounter](https://www.goatcounter.com/), a privacy-first analytics service that avoids tracking personal data or using cookies. Integration is accomplished by including the GoatCounter script in [`layouts/partials/footer.html`](layouts/partials/footer.html). This lightweight JavaScript snippet anonymously records page views, providing insight into site traffic and popular content while respecting user privacy. Analytics are accessible via the GoatCounter dashboard and are used exclusively to understand usage patterns.

I considered displaying page view counts directly in post metadata, but encountered several technical challenges:
- GoatCounter’s JSON API is not accessible from frontend JavaScript due to CORS restrictions; browser requests to the API result in errors because the required `Access-Control-Allow-Origin` header is missing. The API is designed for server-side use only.
- GoatCounter’s `visit_count()` widget offers limited customization and is difficult to style with CSS. Its JSON extension and iframe-based approaches also suffer from CORS limitations and cannot be easily integrated into the frontend.

### Comments via GitHub Discussions

Comments are enabled via [GitHub Discussions](https://github.com/diegoscarabelli/diegoscarabelli.github.io/discussions), using the [Giscus](https://giscus.app/) integration. Each post has a linked discussion thread in the [Blog Comments category](https://github.com/diegoscarabelli/diegoscarabelli.github.io/discussions/categories/blog-comments). When a user visits a post, the Giscus widget loads the relevant discussion, allowing readers to comment using their GitHub account. Configuration is in [`hugo.toml`](hugo.toml) under `[params.comment.giscus]`.

### Subscription via Buttondown

Email subscriptions are managed with [Buttondown](https://buttondown.email/), a newsletter service. The subscription form is embedded using a shortcode ([`layouts/shortcodes/subscribe.html`](layouts/shortcodes/subscribe.html)) and a partial ([`layouts/partials/subscribe.html`](layouts/partials/subscribe.html)). When a user submits their email, it is sent to Buttondown, which manages the mailing list and sends notifications for new posts. Configuration is in [`hugo.toml`](hugo.toml) under `[params.subscription]`.

### Deploying Jupyter Notebooks

Posts authored as Jupyter notebooks are converted to Markdown using `jupyter nbconvert --to markdown`, allowing Hugo to process them as standard content. This workflow supports both static SVG plots and interactive Plotly visualizations.

**How It Works:**

Hugo requires Markdown content, so Jupyter notebooks must be converted using `jupyter nbconvert --to markdown`. During this conversion, `nbconvert` extracts plot outputs from notebook cells as image files.

You have two deployment options:

**Static SVG plots** provide a simple workflow: set `pio.renderers.default = "svg"` in the notebook, then `nbconvert` automatically generates SVG images that Hugo serves as page bundle resources.

**Interactive HTML plots** preserve full Plotly functionality (zoom, pan, hover tooltips) by exporting plots as standalone HTML files before conversion. These files are placed in the `static/` directory where Hugo serves them unchanged. The [`plotly` shortcode](layouts/shortcodes/plotly.html) embeds each HTML file in an iframe, enabling full interactivity without additional JavaScript frameworks or build complexity. This approach keeps files lightweight by loading Plotly.js from a CDN rather than embedding the ~3MB library in each file, while maintaining your standard Jupyter development workflow.

**Notebook Setup for Interactive Plots:**

Add global configuration in the imports cell:
```python
import plotly.io as pio

# Configure Plotly to render as SVG in notebook (for nbconvert compatibility)
pio.renderers.default = "svg"

# Global counter for plot naming
plot_counter = 0

# Global flag to control HTML plot saving
SAVE_HTML_PLOTS = True
```

Define the `save_html_plot()` helper function:
```python
def save_html_plot(fig, output_dir: str = "index_files") -> str:
    """Save a Plotly figure as an HTML file with auto-generated name."""

    global plot_counter
    
    if not SAVE_HTML_PLOTS:
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plot_counter += 1
    plot_name = f"plot_{plot_counter}"
    html_file = f"{output_dir}/{plot_name}.html"
    
    fig.write_html(
        html_file,
        config={'displayModeBar': True, 'responsive': True},
        include_plotlyjs='cdn'
    )
    
    return html_file
```

Call `save_html_plot(fig)` after creating each Plotly figure.

**Deployment Steps for Interactive Plots:**

1. **Execute notebook:** Run all cells in Jupyter to generate HTML plot files in `index_files/`.
2. **Run deployment script:**
   ```bash
   python scripts/deploy_notebook.py <post-name>
   ```
   The script automates the deployment process by:
   - Converting the notebook to Markdown using `nbconvert`.
   - Copying HTML plot files from `content/posts/<post-name>/index_files/` to `static/posts/<post-name>/index_files/`.
   - Replacing SVG image references with Plotly shortcodes in the generated Markdown.

3. **Test locally:**
   ```bash
   hugo server -D
   ```
4. **Commit and push:** GitHub Actions deploys automatically.

**Alternative: Static SVG Deployment**

For simpler deployment without interactivity:

1. Set `SAVE_HTML_PLOTS = False` in notebook imports.
2. Execute notebook cells.
3. Run `jupyter nbconvert --to markdown index.ipynb`.
4. Deploy normally. SVG images are automatically generated and served as page bundle resources.

**Technical Notes:**
- Hugo page bundles serve SVG/image files as resources; HTML files require `static/` directory
- The [`plotly.html` shortcode](layouts/shortcodes/plotly.html) uses responsive iframes
- Plotly.js loads via CDN (Content Delivery Network) to keep HTML files small; the library is fetched from a public server rather than embedded in each file
- Plot names are auto-generated as `plot_1.html`, `plot_2.html`, etc.

## Directory Structure

- [`archetypes/`](archetypes/): Stores archetype templates that define the default front matter and structure for new content types. For example, [`default.md`](archetypes/default.md) provides a template used when running `hugo new` to create a new post or page, ensuring consistency and saving time when authoring new content.
- [`assets/`](assets/): Stores asset files such as CSS, JS, or images to be processed by Hugo’s asset pipeline.
- [`config/`](config/): (Currently empty) Intended for Hugo configuration files, e.g., per-environment settings.
- [`content/`](content/): Main content directory containing all site pages and posts.
    - [`_index.md`](content/_index.md): Defines the homepage content and metadata, including a brief introduction and a list of featured or recent posts.
    - [`posts/`](content/posts/): Contains all blog post Markdown files. Each file represents a single post and includes front matter (title, date, tags) and the main article content.
    - [`about.md`](content/about.md): Provides information about the author and the purpose of the site.
    - [`subscribe.md`](content/subscribe.md): Contains a subscription form for users to sign up for email and RSS updates.
    - [`resources/`](content/resources/): Curated resources, including book recommendations with personal reviews and a selection of noteworthy YouTube channels.
- [`data/`](data/): (Currently empty) For custom data files (YAML, JSON, TOML) used by Hugo.
- [`i18n/`](i18n/): (Currently empty) For translation files to support multilingual content. If present, files in this folder override those in the LoveIt theme’s own `i18n/` directory, allowing you to customize or extend translations for your site.
- [`layouts/`](layouts/): Custom Hugo templates and partials that override or extend the theme’s own `layouts/` files. Hugo uses files in this folder in preference to those in [`themes/LoveIt/layouts/`](themes/LoveIt/layouts/), allowing you to customize the site’s appearance and behavior without modifying the theme directly.
    - [`_default/`](layouts/_default/): Default templates for single pages (`single.html`), list pages (`list.html`), and base templates (`baseof.html`). For example, `single.html` controls the layout for generic single content pages (e.g., `/about/`), while `list.html` is used for section and taxonomy list pages, `section.html` can define how all posts in a section are listed, including custom headers or summaries.
    - [`partials/`](layouts/partials/): Reusable template snippets included in other templates, such as [`footer.html`](layouts/partials/footer.html) (site footer), [`comment.html`](layouts/partials/comment.html) (comments section), and [`subscribe.html`](layouts/partials/subscribe.html) (subscription form). These can be included in any template using Hugo’s `partial` function.
    - [`posts/`](layouts/posts/): Section-specific templates for blog posts. For example, [`single.html`](layouts/posts/single.html) controls the layout for individual blog posts (`/posts/my-post/`), and can include custom logic such as displaying view counts.
    - [`resources/`](layouts/resources/): Custom templates for the `resources` section, if present.
    - [`shortcodes/`](layouts/shortcodes/): Custom Hugo shortcodes, which are reusable snippets invoked in Markdown content using `{{< shortcode >}}` syntax. For example, a `subscribe.html` shortcode can embed a subscription form anywhere in your content.
    - [`taxonomy/`](layouts/taxonomy/): Templates for taxonomy pages (e.g., tags and categories). `terms.html` controls the list of all terms for a taxonomy (e.g., `/tags/` shows all tags), while `term.html` controls the list of content for a specific term (e.g., `/tags/hugo/` shows all posts tagged "hugo").
- [`public/`](public/): The output directory for the generated static site (auto-generated, do not edit directly).
- [`resources/`](resources/): Hugo’s cache and generated files (auto-generated).
- [`static/`](static/): Static files (images, icons, etc.) served as-is, e.g., favicons and site images.
- [`themes/`](themes/): Hugo themes, e.g., the [`LoveIt`](themes/LoveIt/) theme used by this site.

## Special Files

- [`CNAME`](CNAME): Specifies the custom domain for GitHub Pages (`www.nospoonlabs.com`).
- [`hugo.toml`](hugo.toml): Main Hugo configuration file, sets site metadata, theme, menus, and other settings.