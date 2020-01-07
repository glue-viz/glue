Installing with pip
===================

**Platforms:** MacOS X, Linux, and Windows

You can install glue along with **all** :ref:`required and optional dependencies
<glue-deps>` with `pip <https://pip.pypa.io/en/stable/>`__ using::

    pip install glueviz[all,qt]

The above will include domain-specific plugins as well as the PyQt5 library.
If you only want to install glue with all required and only non-domain-specific
optional dependencies (for example excluding the optional dependencies for
astronomy), you can do::

    pip install glueviz[recommended,qt]

And finally, if you don't want to install optional dependencies at all, and
already have PyQt5 or PySide2 installed, you can do::

    pip install glueviz

Note that this will still installed required dependencies.
