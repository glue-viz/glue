# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Glue"
copyright = "2012-2023, Chris Beaumont, Thomas Robitaille, Michelle Borkin"
author = "Chris Beaumont, Thomas Robitaille, Michelle Borkin"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autoclass_content = "both"

nitpick_ignore = [
    ("py:class", "glue.viewers.histogram.layer_artist.HistogramLayerBase"),
    ("py:class", "glue.viewers.scatter.layer_artist.ScatterLayerBase"),
    ("py:class", "glue.viewers.image.layer_artist.ImageLayerBase"),
    ("py:class", "glue.viewers.image.layer_artist.RGBImageLayerBase"),
    ("py:class", "glue.viewers.image.state.BaseImageLayerState"),
    ("py:class", "glue.viewers.common.stretch_state_mixin.StretchStateMixin")
]

viewcode_follow_imported_members = False

numpydoc_show_class_members = False
autosummary_generate = True
automodapi_toctreedirnm = "api"

linkcheck_ignore = [r"https://s3.amazonaws.com"]
linkcheck_retries = 5
linkcheck_timeout = 10


intersphinx_mapping = {
    "python": ("https://docs.python.org/3.7", None),
    "matplotlib": ("https://matplotlib.org", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "echo": ("https://echo.readthedocs.io/en/latest/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_theme_options = {'navigation_with_keys': False}
