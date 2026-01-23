# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

"""Sphinx configuration for AWESPA documentation."""

import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# Add the project root to the path for autodoc to find modules
projectRoot = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(projectRoot / "src"))

# -- Project information -----------------------------------------------------
project = "AWESPA"
copyright = "2024, Airborne Wind Energy Research Group, TU Delft"
author = "Joren Bredael"
release = "1.0.0"
version = "1.0"

# Project description for meta tags
description = "Airborne Wind Energy System Performance Assessment Toolchain"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# Templates path
templates_path = ["_templates"]

# Patterns to exclude from source files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The master toctree document
master_doc = "index"

# Support both RST and Markdown
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"

# Theme options for furo
html_theme_options = {
    "navigation_with_keys": True,
}

# Static files path
html_static_path = ["_static"]

# HTML context for templates
html_context = {
    "display_github": True,
    "github_user": "awegroup",
    "github_repo": "AWESPA",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# -- Autodoc settings --------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}


