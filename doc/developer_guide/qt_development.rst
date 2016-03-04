Qt development in Glue
======================

.. _qthelpers:

Using qt-helpers
----------------

If you are interested in working on some of the Qt-specific code, it's
important that you don't import any code directly from PyQt4, PyQt5, or PySide.
Since we want to maintain backward-compatibility with all of these, you should
always import from ``glue.external.qt`` as if this was one of the Python Qt
packages. For instance, instead of::

    from PyQt4 import QtGui

you should do::

    from glue.external.qt import QtGui

Note that for now, if the PyQt4 and PyQt5 import paths would be different, you
should use the PyQt4 one, and we have provided patches in ``glue.external.qt``
to make PyQt5 backward-compatible.
