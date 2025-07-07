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

### Page Views with Busuanzi
Page view counts are displayed using [Busuanzi](http://busuanzi.ibruce.info/), a free, privacy-friendly service that tracks and displays page and site view counts. The integration is done by including the Busuanzi script in [`layouts/partials/footer.html`](layouts/partials/footer.html) and displaying the view count in [`layouts/posts/single.html`](layouts/posts/single.html). Busuanzi works by running a JavaScript script in each visitor's browser (client side). When a user loads a page, the script sends a request to Busuanzi's server, which increments and stores the view count for that specific page. The server then returns the updated total, which is displayed on the page. This mechanism means that all users' views are aggregated and recorded on Busuanzi's server, but the counting is triggered by each user's browser when they visit the page.

### Comments via GitHub Discussions
Comments are enabled via [GitHub Discussions](https://github.com/diegoscarabelli/diegoscarabelli.github.io/discussions), using the [Giscus](https://giscus.app/) integration. Each post has a linked discussion thread in the [Blog Comments category](https://github.com/diegoscarabelli/diegoscarabelli.github.io/discussions/categories/blog-comments). When a user visits a post, the Giscus widget loads the relevant discussion, allowing readers to comment using their GitHub account. Configuration is in [`hugo.toml`](hugo.toml) under `[params.comment.giscus]`.

### Subscription via Buttondown
Email subscriptions are managed with [Buttondown](https://buttondown.email/), a newsletter service. The subscription form is embedded using a shortcode ([`layouts/shortcodes/subscribe.html`](layouts/shortcodes/subscribe.html)) and a partial ([`layouts/partials/subscribe.html`](layouts/partials/subscribe.html)). When a user submits their email, it is sent to Buttondown, which manages the mailing list and sends notifications for new posts. Configuration is in [`hugo.toml`](hugo.toml) under `[params.subscription]`.

## Directory Structure

- [`archetypes/`](archetypes/): Stores archetype templates that define the default front matter and structure for new content types. For example, [`default.md`](archetypes/default.md) provides a template used when running `hugo new` to create a new post or page, ensuring consistency and saving time when authoring new content.
- [`assets/`](assets/): Stores asset files such as CSS, JS, or images to be processed by Hugo’s asset pipeline.
- [`config/`](config/): (Currently empty) Intended for Hugo configuration files, e.g., per-environment settings.
- [`content/`](content/): Main content directory containing all site pages and posts.
    - [`_index.md`](content/_index.md): Defines the homepage content and metadata, such as the site introduction, featured posts, or custom sections shown on the landing page.
    - [`posts/`](content/posts/): Directory for all blog posts, each as a separate Markdown file. Posts typically include front matter (title, date, tags, etc.) and the main article content. Hugo uses this directory to generate individual blog post pages and blog indexes.
    - [`about.md`](content/about.md): Contains the About page, describing the site's purpose, author background, and any relevant information for visitors.
    - [`subscribe.md`](content/subscribe.md): Provides subscription instructions and embeds the Buttondown email subscription form, allowing users to sign up for updates.
    - [`resources/`](content/resources/): Contains additional resources, curated link collections, or reference materials relevant to the blog's topics.
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