#!/usr/bin/env python3
"""
Automate deployment of Jupyter notebooks with interactive Plotly plots to Hugo.

Workflow:
1. Execute all notebook cells to generate HTML plot files in index_files/
2. Run this script to:
   - Convert notebook to Markdown using nbconvert
   - Copy HTML files from content/posts/<post-name>/index_files/ to static/posts/<post-name>/index_files/
   - Replace SVG image references with Plotly shortcodes in the markdown
3. Test locally with: hugo server -D
4. Commit and push

Usage:
    python scripts/deploy_notebook.py <post-name>
    
Example:
    python scripts/deploy_notebook.py us_deficit_inflation
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import fire


def deploy_notebook(post_name: str):
    """
    Deploy a Jupyter notebook with interactive plots to Hugo.
    
    :param post_name: Name of the post directory (e.g., 'us_deficit_inflation')
    :returns: None
    """
    
    # Define paths
    repo_root = Path(__file__).parent.parent
    post_dir = repo_root / "content" / "posts" / post_name
    notebook_path = post_dir / "index.ipynb"
    markdown_path = post_dir / "index.md"
    html_source_dir = post_dir / "index_files"
    static_target_dir = repo_root / "static" / "posts" / post_name / "index_files"
    
    # Validate paths
    if not post_dir.exists():
        print(f"❌ Error: Post directory not found: {post_dir}")
        sys.exit(1)
    
    if not notebook_path.exists():
        print(f"❌ Error: Notebook not found: {notebook_path}")
        sys.exit(1)
    
    print(f"🚀 Deploying notebook: {post_name}")
    print(f"   Notebook: {notebook_path}")
    print(f"   Output: {markdown_path}")
    
    # Step 1: Convert notebook to markdown
    print("\n📝 Step 1: Converting notebook to markdown...")
    try:
        subprocess.run(
            ["jupyter", "nbconvert", "--to", "markdown", str(notebook_path)],
            cwd=post_dir,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"   ✅ Created: {markdown_path}")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Conversion failed: {e.stderr}")
        sys.exit(1)
    
    # Step 2: Copy HTML files to static directory
    print("\n📂 Step 2: Copying HTML plot files...")
    if html_source_dir.exists():
        html_files = list(html_source_dir.glob("*.html"))
        if html_files:
            static_target_dir.mkdir(parents=True, exist_ok=True)
            for html_file in html_files:
                target_file = static_target_dir / html_file.name
                shutil.copy2(html_file, target_file)
                print(f"   ✅ Copied: {html_file.name}")
            print(f"   📁 Destination: {static_target_dir}")
        else:
            print("   ⚠️  No HTML files found to copy")
    else:
        print(f"   ⚠️  HTML directory not found: {html_source_dir}")
    
    # Step 3: Replace SVG references with Plotly shortcodes
    print("\n🔄 Step 3: Replacing SVG references with Plotly shortcodes...")
    if not markdown_path.exists():
        print("   ❌ Markdown file not found")
        sys.exit(1)
        
    with open(markdown_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match SVG image references: ![svg](index_files/index_X_Y.svg)
    svg_pattern = re.compile(r'!\[svg\]\(index_files/index_\d+_\d+\.svg\)')
    svg_matches = svg_pattern.findall(content)
    
    if svg_matches:
        print(f"   Found {len(svg_matches)} SVG references")
        # Replace each SVG with corresponding HTML plot, validating that the HTML file exists
        for i, match in enumerate(svg_matches, 1):
            html_filename = f"plot_{i}.html"
            source_html_path = html_source_dir / html_filename
            static_html_path = static_target_dir / html_filename
            
            if not source_html_path.exists() and not static_html_path.exists():
                print(f"   ⚠️  Skipping replacement for {match}")
                print(f"      → Expected HTML file not found: {source_html_path} or {static_html_path}")
                continue
            
            shortcode = f'{{{{< plotly file="index_files/{html_filename}" >}}}}'
            content = content.replace(match, shortcode, 1)
            print(f"   ✅ Replaced {match}")
            print(f"      → {shortcode}")
        
        # Write updated content
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n   💾 Updated: {markdown_path}")
    else:
        print("   ℹ️  No SVG references found")
    
    # Summary
    print("\n" + "="*70)
    print("✅ Deployment complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Test locally:")
    print("   hugo server -D")
    print("2. Commit and push:")
    print("   git add .")
    print(f"   git commit -m 'Deploy notebook: {post_name}'")
    print("   git push")
    print()


if __name__ == "__main__":
    fire.Fire(deploy_notebook)
