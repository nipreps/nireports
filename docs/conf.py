# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
try:
    # This option is only available with Python 3.8
    from importlib.metadata import version as get_version
except ImportError:
    from importlib_metadata import version as get_version

# -- Project information -----------------------------------------------------

project = "nireports"
copyright = "2023, The NiPreps Developers"
author = "The NiPreps Developers"

# The full version, including alpha/beta/rc tags
release = get_version("nireports")


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinxcontrib.apidoc",
    "nipype.sphinxext.apidoc",
]

autodoc_mock_imports = [
    "nilearn",
    "nitime",
    "pandas",
    "seaborn",
    "skimage",
    "svgutils",
    "templateflow",
    "transforms3d",
    "yaml",
]

# Accept custom section names to be parsed for numpy-style docstrings
# of parameters.
# Requires pinning sphinxcontrib-napoleon to a specific commit while
# https://github.com/sphinx-contrib/napoleon/pull/10 is merged.
napoleon_use_param = False
napoleon_custom_sections = [
    ("Inputs", "Parameters"),
    ("Outputs", "Parameters"),
    ("Attributes", "Parameters"),
    ("Mandatory Inputs", "Parameters"),
    ("Optional Inputs", "Parameters"),
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_js_files = [
    "js/version-switch.js",
]
html_css_files = [
    "css/version-switch.css",
]

# Solve ReadTheDocs build error (expects master file to be contents.rst)
master_doc = "index"

# -- Extension configuration -------------------------------------------------

apidoc_module_dir = "../nireports"
apidoc_output_dir = "api"
apidoc_excluded_paths = ["conftest.py", "*/tests/*", "tests/*", "data/*", "testing.py"]
apidoc_separate_modules = True
apidoc_extra_args = ["--module-first", "-d 1", "-T"]

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "bids": ("https://bids-standard.github.io/pybids/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
    "nipype": ("https://nipype.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "smriprep": ("https://www.nipreps.org/smriprep/", None),
    "surfplot": ("https://surfplot.readthedocs.io/en/latest/", None),
    "templateflow": ("https://www.templateflow.org/python-client", None),
}

# -- Options for versioning extension ----------------------------------------
scv_show_banner = True
