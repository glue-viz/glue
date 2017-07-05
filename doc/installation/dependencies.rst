.. _glue-deps:

Full list of dependencies
=========================

Glue has the following required dependencies:

* Python 2.7, or 3.3 and higher
* `Numpy <http://www.numpy.org>`_ 1.9 or later
* `Matplotlib <http://www.matplotlib.org>`_ 1.4 or later
* `Pandas <http://pandas.pydata.org/>`_ 0.14 or later
* `Astropy <http://www.astropy.org>`_ 1.0 or higher
* `setuptools <http://setuptools.readthedocs.io/en/latest/>`_ 1.0 or later
* Either `PySide <http://pyside.org>`__ or `PyQt
  <https://riverbankcomputing.com/software/pyqt/intro>`__ (both PyQt4 and PyQt5 are supported)
* `QtPy <https://pypi.python.org/pypi/QtPy/>`__ 1.1.1 or higher - this is an
  abstraction layer for the Python Qt packages
* `IPython <http://ipython.org>`_ 4.0 or higher
* `ipykernel <https://pypi.python.org/pypi/ipykernel>`_
* `qtconsole <http://jupyter.org/qtconsole/>`_
* `dill <http://pythonhosted.org/dill/>`_ 0.2 or later (which improves session saving)
* `h5py <http://www.h5py.org>`_ 2.4 or later, for reading HDF5 files
* `xlrd <https://pypi.python.org/pypi/xlrd>`_ 1.0 or later, for reading Excel files
* `glue-vispy-viewers <https://pypi.python.org/pypi/glue-vispy-viewers>`_, which provide 3D viewers

The following optional dependencies are also highly recommended and
domain-independent:

* `SciPy <http://www.scipy.org>`_
* `scikit-image <http://scikit-image.org>`_
* `plotly <https://plot.ly>`_ for exporting to plot.ly

Finally, there are domain-specific optional dependencies. For astronomy, these
are:

* `astrodendro <http://dendrograms.org>`_ for dendrograms
* `pyavm <https://astrofrog.github.io/pyavm/>`_ for reading AVM metadata
* `spectral-cube <http://spectral-cube.readthedocs.io>`_ for reading spectral cubes

You can check which dependencies are installed and which versions are available
by running (once glue is installed)::

    glue-deps list

It is also possible to install missing dependencies with::

    glue-deps install
