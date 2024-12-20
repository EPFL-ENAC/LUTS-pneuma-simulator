# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

import pNeuma_simulator as ps

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pNeuma-simulator"
copyright = "2024, Georg Anagnostopoulos"
author = "Georg Anagnostopoulos"
__version__ = ps.__version__
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "myst_parser",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    ".github",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "README.md",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}


# -- Automatically run apidoc to generate rst from code
# https://github.com/readthedocs/readthedocs.org/issues/1139
def run_apidoc(_):
    from sphinx.ext.apidoc import main

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    cur_dir = os.path.abspath(os.path.dirname(__file__))

    for module_dir in ["pNeuma_simulator"]:
        module = os.path.join(cur_dir, "..", module_dir)
        output = os.path.join(cur_dir, "auto_source", module_dir)
        main(["-e", "-f", "-o", output, module])


def setup(app):
    app.connect("builder-inited", run_apidoc)
