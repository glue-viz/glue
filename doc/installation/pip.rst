Installing with pip
===================

**Platforms:** MacOS X, Linux, and Windows

You can install glue along with **all** :ref:`required and optional dependencies
<glue-deps>` with `pip <https://pip.pypa.io/en/stable/>`__ using::

    pip install glueviz[all]

The above will include domain-specific plugins. If you only want to install glue
with all required and only non-domain-specific optional dependencies (for
example excluding the optional dependencies for astronomy), you can do::

    pip install glueviz[recommended]

And finally, if you don't want to install optional dependencies at all::

    pip install glueviz

Note that this will still installed required dependencies.

If you are using Python 2.7, you will need to also make sure you install Qt and
either `PyQt5 <https://riverbankcomputing.com/software/pyqt/intro>`__ or `PySide2
<https://wiki.qt.io/Qt_for_Python>`__, since these cannot be automatically installed
with the ``pip`` command unless you are using Python 3.x. See the section on
:ref:`installing-qt` for more details.
