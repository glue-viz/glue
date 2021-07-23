.. _known-issues:

Known issues and solutions
==========================

.. _apple-m1:

Using glue on Apple M1 hardware
-------------------------------

At the moment, glue is not easy to set up to run using a Python stack running
natively on M1 processors - instead you should
`enable Rosetta 2 <https://support.apple.com/en-gb/HT211861>`_
for the **Terminal.app**. You can do this by selecting **Terminal.app** in the
Finder, then going to **File** and selecting **Get Info**, then clicking the
checkbox **Open using Rosetta**.

Once you have done this, follow the regular
:ref:`installation` instructions for glue as if you were using an intel Mac.
Be sure to also read `Using glue on Big Sur`_ since this is the version of
MacOS X that is installed on Apple M1 hardware.

.. _apple-bigsur:

Using glue on Big Sur
---------------------

If you have issues with getting the 3D functionality to work, you will need to
install the latest development version of VisPy. If you are using conda, you
will first need to remove the conda-installed version of VisPy using::

    conda remove vispy --force

You can then install the latest development version of VisPy with::

    pip install git+https://github.com/vispy/vispy

Once this is done the 3D viewers should work properly.

3D viewers not working on Linux with PyQt5
------------------------------------------

Until recently, the main conda packages for PyQt5 provided by Anaconda did not
support OpenGL, which is needed for the 3D viewers. However, the latest
available conda packages now properly support OpenGL, so if you are having
issues getting the 3D viewers to work on Linux, try the following: first, make
sure you have the latest version of conda installed::

    conda update -n root conda

then update all packages in your environment using::

    conda update -c glueviz --all

Updating all packages is safest to make sure there are no conflicts between
packages, but if you prefer to try updating just the relevant packages, you
can try::

    conda update qt pyqt icu sip

but note that this may not always be sufficient to fix the issue.

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

Undefined symbol: ``_ZNK7QSslKey9algorithmEv``
----------------------------------------------

On certain Linux installations, when using Anaconda/conda to manage the Python
installation you are using for glue, you may run into the following error when
launching glue::

    ImportError: /usr/lib/libkdecore.so.5: undefined symbol: _ZNK7QSslKey9algorithmEv

This should be resolved in recent versions of the PyQt conda package, so
updating to the latest version should be sufficient to resolve this issue.
First, make sure you have the latest version of conda installed::

    conda update -n root conda

then update all packages in your environment using::

    conda update -c glueviz --all

Updating all packages is safest to make sure there are no conflicts between
packages, but if you prefer to try updating just the relevant packages, you
can try::

    conda update qt pyqt icu sip

but note that this may not always be sufficient to fix the issue.

Incompatibility with PySide2 5.12.0 and 5.12.1
----------------------------------------------

Glue is known to not work correctly with PySide2 5.12.0 due to `a bug in PySide2
<https://bugreports.qt.io/browse/PYSIDE-883>`_ and with PySide2 5.12.1 due to `a
different bug in PySide2 <https://bugreports.qt.io/browse/PYSIDE-937>`_. If you
are using PySide2, be sure to use 5.12.2 or later.
