.. _installing-qt:

Installing PyQt or PySide
=========================

.. note:: If you are installing glue with conda or with pip then PyQt5 should be
          automatically installed so you can ignore this page. If however you
          need to manually install PyQt5 or PySide2, then read on!

If you are using Linux, PyQt and PySide will typically be available in the
built-in package manager. For example, if you are using Ubuntu, then you can do::

    sudo apt-get install python3-pyqt5

If you are using MacOS X, then if you are using MacPorts to manage your Python
installation, you can do::

    sudo port install py37-pyqt5

assuming you are using Python 3.7 (modify the ``py37`` version as needed).
