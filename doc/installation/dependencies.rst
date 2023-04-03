.. _glue-deps:

Full list of dependencies
=========================

Glue has the following required dependencies:

* Python 3.8 or later
* `Numpy <https://www.numpy.org>`_ 1.16 or later
* `Matplotlib <https://matplotlib.org/>`_ 3.2 or later
* `SciPy <https://www.scipy.org>`_ 1.0 or later
* `Pandas <https://pandas.pydata.org/>`_ 1.0 or later
* `Astropy <https://www.astropy.org>`_ 4.0 or higher
* `setuptools <https://setuptools.readthedocs.io>`_ 30.3 or later
* Either `PyQt5 <https://www.riverbankcomputing.com/software/pyqt/intro>`__ or
  `PySide2 <https://wiki.qt.io/PySide2>`__ 5.14 or later
* `QtPy <https://pypi.org/project/QtPy/>`__ 1.9 or later - this is an
  abstraction layer for the Python Qt packages
* `IPython <https://ipython.org>`_ 4.0 or later
* `ipykernel <https://pypi.org/project/ipykernel>`_ 4.0 or later
* `qtconsole <https://jupyter.org/qtconsole/>`_
* `dill <https://pypi.org/project/dill>`_ 0.2 or later (which improves session saving)
* `h5py <https://www.h5py.org>`_ 2.10 or later, for reading HDF5 files
* `xlrd <https://pypi.org/project/xlrd>`_ 1.2 or later, for reading Excel files
* `mpl-scatter-density <https://github.com/astrofrog/mpl-scatter-density>`_ 0.7 or later, for making
  scatter density maps of many points.

The following optional dependencies are also highly recommended and
domain-independent:

* `SciPy <https://www.scipy.org>`_
* `scikit-image <https://scikit-image.org>`_

Finally, there are domain-specific optional dependencies. For astronomy, these
are:

* `astrodendro <https://dendrograms.readthedocs.io>`_ for dendrograms
* `pyavm <https://astrofrog.github.io/pyavm/>`_ for reading AVM metadata
* `spectral-cube <https://spectral-cube.readthedocs.io>`_ for reading spectral cubes

You can check which dependencies are installed and which versions are available
by running (once glue is installed)::

    glue-deps list
