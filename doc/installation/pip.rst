Installing with pip
===================

**Platforms:** MacOS X, Linux, and Windows

Installing glue with `pip <https://pip.pypa.io>`__ is possible, although you
will need to first make sure that you install Qt and either `PyQt
<https://riverbankcomputing.com/software/pyqt/intro>`__ or `PySide
<http://pyside.org>`__, since these cannot be automatically installed with the
``pip`` command. See the section on :ref:`installing-qt` for more details.

Assuming that you have either PyQt or PySide installed, you can install glue
along with **all** :ref:`required and optional dependencies <glue-deps>` using::

    pip install glueviz[all]

The above will include domain-specific plugins. If you only want to install glue
with all required and only non-domain-specific optional dependencies (for
example excluding the optional dependencies for astronomy), you can do::

    pip install glueviz[recommended]

And finally, if you don't want to install optional dependencies at all::

    pip install glueviz

Note that this will still installed required dependencies.
