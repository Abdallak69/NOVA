# Configuration file for the Sphinx documentation builder.

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# -- Project information -----------------------------------------------------

project = "NOVA - Quantum Neural Network Framework"
author = "NOVA Team"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

myst_enable_extensions = [
    "colon_fence",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}
