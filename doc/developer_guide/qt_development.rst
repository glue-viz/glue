Qt development in Glue
======================

.. _qtpy:

Using QtPy
----------

If you are interested in working on some of the Qt-specific code, it's
important that you don't import any code directly from PyQt or PySide.
Since we want to maintain backward-compatibility with all of these, you should
always use the `QtPy <https://pypi.org/project/QtPy>`__ package. The way to
use this package is to import from the ``qtpy`` module as if it was the
``PyQt5`` module, and QtPy will automatically translate this into the
appropriate imports for PyQt or PySide if needed. For instance, instead of::

    from PyQt5 import QtCore

you should do::

    from qtpy import QtCore
