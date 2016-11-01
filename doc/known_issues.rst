.. _known-issues:

Known issues
============

Qt internal error: qt_menu.nib could not be loaded
--------------------------------------------------

When using Glue with the `PySide <https://wiki.qt.io/PySide>`_ bindings, the
following error sometimes occurs for some MacOS X conda users::

    Qt internal error: qt_menu.nib could not be loaded. The .nib file should be
    placed in QtGui.framework/Versions/Current/Resources/  or in the resources
    directory of your application bundle.

This is due to the PySide conda package in the ``defaults`` conda channel being
broken (see the following
`issue <https://github.com/ContinuumIO/anaconda-issues/issues/1132>`_ for the
latest status on this issue). The workaround is to either use PyQt instead of
PySide, or to use the PySide package from the ``conda-forge`` channel::

    conda install -c conda-forge pyside

Undefined symbol: _ZNK7QSslKey9algorithmEv
------------------------------------------

On certain Linux installations, when using Anaconda/conda to manage the Python
installation you are using for glue, you may run into the following error when
launching glue::

    ImportError: /usr/lib/libkdecore.so.5: undefined symbol: _ZNK7QSslKey9algorithmEv

This should be resolved in recent versions of the PyQt conda package, so
updating to the latest version should be sufficient to resolve this issue::

    conda install pyqt
