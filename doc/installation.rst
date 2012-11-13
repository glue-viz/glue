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
 * `Matplotlib <http://www.matplotlib.org>`_ 1.1.0 or later

*Optional*

* `Scipy <http://www.scipy.org>`_. Used for some analysis features
* `PyQt4 <http://www.riverbankcomputing.co.uk/software/pyqt/download>`_
   4.9.4 or later (which requires
   `SIP <http://www.riverbankcomputing.co.uk/software/sip/download>`_ ). Required for the graphical user interface.
* `IPython <http://www.ipython.org>`_ For using the IPython terminal within the GUI.

*Optional, for Astronomy*

* `Astropy <http://www.astropy.org>`_
* `ATpy <http://atpy.github.com>`_
* `h5py <http://code.google.com/p/h5py/>`_

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
    pip install vo
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
    pip install vo
    pip install atpy
