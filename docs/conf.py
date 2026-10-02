from __future__ import annotations

import os
import sys
from datetime import datetime

# Avoid Matplotlib trying to write to $HOME during autodoc imports (RTD/build environments are often read-only).
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

PROJECT_ROOT = os.path.abspath("..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

project = "Exposure Inequality Pipeline"
author = "the paper authors"
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

try:
    import furo  # noqa: F401

    html_theme = "furo"
except ModuleNotFoundError:
    html_theme = "alabaster"
html_static_path = ["_static"]

autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
