Qt development in Glue
======================

.. _qtpy:

Using QtPy
----------

If you are interested in working on some of the Qt-specific code, it's
important that you don't import any code directly from PyQt4, PyQt5, or PySide.
Since we want to maintain backward-compatibility with all of these, you should
always use the `QtPy <https://pypi.python.org/pypi/QtPy>`__ package. The way to
use this package is to import from the ``qtpy`` module as if it was the
``PyQt5`` module, and QtPy will automatically translate this into the
appropriate imports for PySide or PyQt4 if needed. For instance, instead of::

    from PyQt4 import QtCore

you should do::

    from qtpy import QtCore

Note that if the PyQt4 and PyQt5 import paths would be different, you should
use the PyQt5 one.
