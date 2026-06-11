"""Sphinx configuration for LSQMultivariateSpline documentation."""

project = "LSQMultivariateSpline"
copyright = "2026"
author = "Nikolas Valsamidis"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

html_theme = "alabaster"
autodoc_typehints = "none"
