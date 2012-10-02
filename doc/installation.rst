Installing Glue
===============

Requirements
------------

This section lists the dependencies required by Glue. If only using ``glue.core``, then the only dependencies are:

* `Python <http://www.python.org>`_ 2.6 or 2.7
* `Numpy <http://numpy.scipy.org>`_ 1.4.0 or later

If using the graphical user interface (GUI), you will need:

* `PyQt4 <http://www.riverbankcomputing.co.uk/software/pyqt/download>`_ 4.9.4 or later (which requires `SIP <http://www.riverbankcomputing.co.uk/software/sip/download>`_)
* `Scipy <http://www.scipy.org>`_
* `Matplotlib <http://www.matplotlib.org>`_
* `IPython <http://www.ipython.org>`_

If you are using Glue for Astronomical purposes, you will also need:

* `Astropy <http://www.astropy.org>`_
* `ATpy <http://atpy.github.com>`_
* `h5py <http://code.google.com/p/h5py/>`_

Linux
-----

Most of the dependencies can be installed on Linux using package managers.

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

Finally, you can install glue with::

    git clone git://github.com/glue-viz/glue.git
    cd glue
    python setup.py install

MacOS X
-------

There are different ways to set up the dependencies on Mac (including a fully manual installation) but we recommend the use of `MacPorts <http://www.macports.org>`_. For information about using MacPorts to manage your Python installation, see `here <http://astrofrog.github.com/macports-python/>`_.

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

Finally, you can install glue with::

    git clone git://github.com/glue-viz/glue.git
    cd glue
    python setup.py install


Development
-----------

For testing/development, you will also need:

* `py.test <http://www.pytest.org>`_
* `mock <http://www.voidspace.org.uk/python/mock/>`_

