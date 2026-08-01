project = "spAlignDE"
author = "spAlignDE team"
copyright = "2026, spAlignDE team"

release = ""
version = ""

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "sticky_navigation": False,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]
html_title = "spAlignDE documentation"
html_baseurl = "https://dsong-lab.github.io/spAlignDE/"

epub_show_urls = "footnote"

# Render notebook outputs as saved in notebooks without executing during docs build.
nb_execution_mode = "off"
