Installing with pip
===================

**Platforms:** MacOS X, Linux, and Windows

Installing glue with `pip <https://pip.pypa.io>`__ is also possible, although you
will need to first make sure that you install Qt and either
`PyQt <https://riverbankcomputing.com/software/pyqt/intro>`_ or
`PySide <http://pyside.org>`_, since these cannot be automatically
installed. See the section on `Installing PyQt or PySide`_

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
