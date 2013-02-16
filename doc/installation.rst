.. _installation:

Installing Glue
===============
The source code for Glue is available on `GitHub <http://www.github.com/glue-viz/glue>`_. This page describes how to install Glue on your machine. Alternatively, you can also try downloading a pre-built package.

Pre-Built Packages
------------------
Mac users (OS X >= 10.7) can download the latest version of Glue `here <https://www.dropbox.com/sh/a7jbvaruzdrri8j/8En3jGR3n6>`_. This is the easiest option, as it includes all of Glue's dependencies.

Building From Source
--------------------
Dependencies
^^^^^^^^^^^^

*Required*

 * `Python <http://www.python.org>`_ 2.6 or 2.7
 * `Numpy <http://numpy.scipy.org>`_ 1.4.0 or later
 * `Matplotlib <http://www.matplotlib.org>`_ 1.1.0 or later (1.2.0 or later recommended, due to an image tinting bug in 1.1.1)
 * `PyQt4 <http://www.riverbankcomputing.co.uk/software/pyqt/download>`_ 4.8.0 or later or `PySide <http://qt-project.org/wiki/PySide>`_ v1.1.0 or later. PyQt4 further requires `SIP <http://www.riverbankcomputing.co.uk/software/sip/download>`_.

.. warning:: PySide support is still experimental. We currently recommend PyQt4.

.. note:: Users with both PyQt4 and PySide can select between bindings by setting the ``QT_API`` environment variable to either ``pyside`` or ``pyqt4``.


*Optional*

* `Scipy <http://www.scipy.org>`_. 0.10.0 or greater. Used for some analysis features
* `IPython <http://www.ipython.org>`_ For using the IPython terminal within the GUI.

*Optional, for Astronomy*

* `Astropy <http://www.astropy.org>`_
* `ATpy <http://atpy.github.com>`_
* `h5py <http://code.google.com/p/h5py/>`_

.. note:: If Astropy is not installed, Glue will fallback to importing the legacy PyFits and PyWCS modules.

*Optional, for development*

* `py.test <http://www.pytest.org>`_
* `mock <http://www.voidspace.org.uk/python/mock/>`_


Installation
^^^^^^^^^^^^

Once Glue's dependencies have been installed (see below), building Glue is straightfoward.

Using git::

    git clone git://github.com/glue-viz/glue.git
    cd glue
    python setup.py install

Or, with pip::

    pip install -e git+git://github.com/glue-viz/glue.git#egg=glue


Ubuntu
^^^^^^

The main dependencies can be installed with::

    sudo apt-get install python-numpy
    sudo apt-get install python-scipy
    sudo apt-get install python-matplotlib
    sudo apt-get install python-qt4
    sudo apt-get install pyqt4-dev-tools
    sudo apt-get install ipython
    sudo apt-get install python-zmq
    sudo apt-get install python-pygments

    sudo apt-get install python-pip

Once these are installed, you can use ``pip`` to install the remaining ones::

    pip install astropy
    pip install -e svn+http://svn6.assembla.com/svn/astrolib/trunk/vo/#egg=vo
    pip install atpy


MacOS X
^^^^^^^

There are different ways to set up the dependencies on Mac (including a fully
manual installation) but we recommend the use of `MacPorts
<http://www.macports.org>`_. For information about using MacPorts to manage
your Python installation, see `here
<http://astrofrog.github.com/macports-python/>`_.

The main dependencies can be installed with::

    sudo port install python27
    sudo port install py27-numpy
    sudo port install py27-scipy
    sudo port install py27-matplotlib
    sudo port install py27-pyqt4
    sudo port install py27-ipython
    sudo port install py27-pip

Once these are installed, you can use ``pip`` to install the remaining ones::

    pip install astropy
    pip install -e svn+http://svn6.assembla.com/svn/astrolib/trunk/vo/#egg=vo
    pip install atpy
    
The Enthought Python Distribution
^^^^^^^^^^^^^^^

The `Enthought Python Distribution <http://www.enthought.com/products/epd.php>`_ contains most of Glue's dependencies. Building Glue on top of EPD involves::

    pip install astropy
    pip install -e svn+http://svn6.assembla.com/svn/astrolib/trunk/vo/#egg=vo
    pip install atpy
    pip install -e git+git://github.com/glue-viz/glue.git#egg=glue 
   

Anaconda
^^^^^^^^
The `Anaconda <https://store.continuum.io/cshop/anaconda>`_ distribution also contains most of Glue's dependencies. Installation instructions are the same as for the Enthought Python Distribution Above.

Running Glue
------------

Installing glue from source will create a executable ``glue`` script
that should be in your path. Running ``glue`` from the command line will
start the program. Glue accepts a variety of command-line
arguments. See ``glue --help`` for examples.

.. note:: On Windows, installation creates a ``glue.bat`` script into the python script directory (e.g., ``C:\Python27\Scripts``). Windows users can create a desktop shortcut for this file, and run Glue by double clicking on the icon.
