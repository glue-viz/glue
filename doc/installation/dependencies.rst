.. _glue-deps:

Full list of dependencies
=========================

Glue has the following required dependencies:

* Python 2.7, or 3.3 and higher
* `Numpy <https://www.numpy.org>`_ 1.9 or later
* `Matplotlib <https://matplotlib.org/>`_ 2.0 or later
* `Pandas <https://pandas.pydata.org/>`_ 0.14 or later
* `Astropy <https://www.astropy.org>`_ 2.0 or higher
* `setuptools <https://setuptools.readthedocs.io>`_ 1.0 or later
* Either `PyQt5 <https://www.riverbankcomputing.com/software/pyqt/intro>`__ or
  `PySide2 <https://wiki.qt.io/PySide2>`__
* `QtPy <https://pypi.org/project/QtPy/>`__ 1.2 or higher - this is an
  abstraction layer for the Python Qt packages
* `IPython <https://ipython.org>`_ 4.0 or higher
* `ipykernel <https://pypi.org/project/ipykernel>`_
* `qtconsole <https://jupyter.org/qtconsole/>`_
* `dill <https://pypi.org/project/dill>`_ 0.2 or later (which improves session saving)
* `h5py <https://www.h5py.org>`_ 2.4 or later, for reading HDF5 files
* `xlrd <https://pypi.org/project/xlrd>`_ 1.0 or later, for reading Excel files
* `mpl-scatter-density <https://github.com/astrofrog/mpl-scatter-density>`_, for making
  scatter density maps of many points.
* `bottleneck <https://pypi.org/project/Bottleneck/>`_, for fast NaN-friendly computations

The following optional dependencies are also highly recommended and
domain-independent:

* `SciPy <https://www.scipy.org>`_
* `scikit-image <https://scikit-image.org>`_
* `plotly <https://plot.ly>`_ for exporting to plot.ly

Finally, there are domain-specific optional dependencies. For astronomy, these
are:

* `astrodendro <https://dendrograms.readthedocs.io>`_ for dendrograms
* `pyavm <https://astrofrog.github.io/pyavm/>`_ for reading AVM metadata
* `spectral-cube <https://spectral-cube.readthedocs.io>`_ for reading spectral cubes

You can check which dependencies are installed and which versions are available
by running (once glue is installed)::

    glue-deps list

It is also possible to install missing dependencies with::

    glue-deps install
