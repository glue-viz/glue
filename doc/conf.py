# -*- coding: utf-8 -*-
#
# Glue documentation build configuration file

import os
import sys
from pkg_resources import get_distribution

# -- General configuration ----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '1.6'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.mathjax',
              'sphinx.ext.viewcode',
              'sphinx.ext.intersphinx',
              'numpydoc',
              'sphinx_automodapi.automodapi',
              'sphinx_automodapi.smart_resolver',
              'sphinxcontrib.spelling']

# Add the redirect.py plugin which is in this directory
sys.path.insert(0, os.path.abspath('.'))
extensions.append('redirect')

# Workaround for RTD where the default encoding is ASCII
if os.environ.get('READTHEDOCS') == 'True':
    import locale
    locale.setlocale(locale.LC_ALL, 'C.UTF-8')

intersphinx_cache_limit = 10     # days to keep the cached inventories
intersphinx_mapping = {
    'python': ('https://docs.python.org/3.7', None),
    'matplotlib': ('https://matplotlib.org', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'astropy': ('http://docs.astropy.org/en/stable/', None),
    'echo': ('https://echo.readthedocs.io/en/latest/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'Glue'
copyright = u'2012-2019, Chris Beaumont, Thomas Robitaille, Michelle Borkin'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
version = release = get_distribution('glue-core').version

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', '_templates', '.eggs']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output --------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
try:  # use ReadTheDocs theme, if installed
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path(), ]
except ImportError:
    pass

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = '_static/logo.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'Gluedoc'

# -- Options for LaTeX output -------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ('index', 'Glue.tex', u'Glue Documentation',
     u'Chris Beaumont, Thomas Robitaille, Michelle Borkin', 'manual'),
]

# -- Options for manual page output -------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'glue', u'Glue Documentation',
     [u'Chris Beaumont, Thomas Robitaille, Michelle Borkin'], 1)
]

# -- Options for Texinfo output -----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    ('index', 'Glue', u'Glue Documentation',
     u'Chris Beaumont, Thomas Robitaille, Michelle Borkin',
     'Glue', 'One line description of project.', 'Miscellaneous'),
]

# -- Additional options------- ------------------------------------------------

todo_include_todos = True
autoclass_content = 'both'

nitpick_ignore = [('py:class', 'object'), ('py:class', 'str'),
                  ('py:class', 'list'), ('py:obj', 'numpy array'),
                  ('py:obj', 'integer'), ('py:obj', 'Callable'),
                  ('py:obj', 'list'),
                  ('py:class', 'PyQt5.QtWidgets.QMainWindow'),
                  ('py:class', 'PyQt5.QtWidgets.QWidget'),
                  ('py:class', 'PyQt5.QtWidgets.QTextEdit'),
                  ('py:class', 'PyQt5.QtWidgets.QTabBar'),
                  ('py:class', 'PyQt5.QtWidgets.QLabel'),
                  ('py:class', 'PyQt5.QtWidgets.QComboBox'),
                  ('py:class', 'PyQt5.QtWidgets.QMessageBox'),
                  ('py:class', 'PyQt5.QtWidgets.QDialog'),
                  ('py:class', 'PyQt5.QtWidgets.QToolBar'),
                  ('py:class', 'PyQt5.QtWidgets.QStyledItemDelegate'),
                  ('py:class', 'PyQt5.QtCore.QMimeData'),
                  ('py:class', 'PyQt5.QtCore.QAbstractListModel'),
                  ('py:class', 'PyQt5.QtCore.QThread'),
                  ('py:class', 'QAction'),
                  ('py:class', 'QByteArray'),
                  ('py:class', 'QCursor'),
                  ('py:class', 'QFont'),
                  ('py:class', 'QFontInfo'),
                  ('py:class', 'QFontMetrics'),
                  ('py:class', 'QGraphicsEffect'),
                  ('py:class', 'QGraphicsProxyWidget'),
                  ('py:class', 'QIcon'),
                  ('py:class', 'QLayout'),
                  ('py:class', 'QLocale'),
                  ('py:class', 'QMargins'),
                  ('py:class', 'QMetaObject'),
                  ('py:class', 'QObject'),
                  ('py:class', 'QPaintEngine'),
                  ('py:class', 'QPainter'),
                  ('py:class', 'QPalette'),
                  ('py:class', 'QPalette.ColorRole'),
                  ('py:class', 'QPixmap'),
                  ('py:class', 'QPoint'),
                  ('py:class', 'QRect'),
                  ('py:class', 'QRegion'),
                  ('py:class', 'QSize'),
                  ('py:class', 'QSizePolicy'),
                  ('py:class', 'QStyle'),
                  ('py:class', 'QToolBar'),
                  ('py:class', 'QThread'),
                  ('py:class', 'QScreen'),
                  ('py:class', 'QWidget'),
                  ('py:class', 'QWidget.RenderFlag'),
                  ('py:class', 'QWidget.RenderFlags'),
                  ('py:class', 'QWindow'),
                  ('py:class', 'Qt.ContextMenuPolicy'),
                  ('py:class', 'Qt.FindChildOption'),
                  ('py:class', 'Qt.FindChildOptions'),
                  ('py:class', 'Qt.FocusPolicy'),
                  ('py:class', 'Qt.InputMethodHints'),
                  ('py:class', 'Qt.LayoutDirection'),
                  ('py:class', 'Qt.Orientation'),
                  ('py:class', 'Qt.TimerType'),
                  ('py:class', 'Qt.ToolBarAreas'),
                  ('py:class', 'Qt.ToolButtonStyle'),
                  ('py:class', 'Qt.WindowFlags'),
                  ('py:class', 'Qt.WindowModality'),
                  ('py:class', 'Qt.WindowStates'),
                  ('py:class', 'Qt.WindowType'),
                  ('py:class', 'Qt.WindowType'),
                  ('py:obj', "str ('file' | 'directory' | 'label')"),
                  ('py:obj', 'function(application)'),
                  ('py:class', 'sip.voidptr'),
                  ('py:obj', 'glue.viewers.common.qt.toolbar.BasicToolbar.insertAction'),
                  ('py:obj', 'glue.viewers.common.qt.toolbar.BasicToolbar.setTabOrder'),
                  ('py:class', 'builtins.object'),
                  ('py:class', 'builtins.list'),
                  ('py:class', 'builtins.type'),
                  ('py:class', 'glue.viewers.histogram.layer_artist.HistogramLayerBase'),
                  ('py:class', 'glue.viewers.scatter.layer_artist.ScatterLayerBase'),
                  ('py:class', 'glue.viewers.image.layer_artist.ImageLayerBase'),
                  ('py:class', 'glue.viewers.image.layer_artist.RGBImageLayerBase'),
                  ('py:mod', 'glue.core'),
                  ('py:mod', 'glue.viewers'),
                  ('py:mod', 'glue.viewers.scatter'),
                  ('py:mod', 'glue.viewers.common'),
                  ('py:mod', 'glue.viewers.common.qt.mouse_mode'),
                  ('py:mod', 'glue.viewers.common.qt.toolbar_mode'),
                  ('py:mod', 'glue.dialogs.custom_component'),
                  ('py:class', 'echo.core.HasCallbackProperties'),
                  ('py:class', 'echo.core.CallbackProperty'),
                  ('py:class', 'echo.selection.SelectionCallbackProperty'),
                  ('py:class', 'glue.viewers.image.state.BaseImageLayerState'),
                  ('py:class', 'glue.viewers.common.qt.data_viewer_with_state.DataViewerWithState')]

# coax Sphinx into treating descriptors as attributes
# see https://bitbucket.org/birkenfeld/sphinx/issue/1254/#comment-7587063
from glue.utils.qt.widget_properties import WidgetProperty
WidgetProperty.__get__ = lambda self, *args, **kwargs: self

viewcode_follow_imported_members = False

numpydoc_show_class_members = False
autosummary_generate = True
automodapi_toctreedirnm = 'api'

linkcheck_ignore = [r'https://s3.amazonaws.com']
linkcheck_retries = 5
linkcheck_timeout = 10
