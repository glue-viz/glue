.. _faq:

Frequently Asked Questions
==========================

What data formats does Glue understand?
---------------------------------------
Glue relies on several libraries to parse different file formats:

 * `Astropy <http://www.astropy.org>`_ for FITS images and tables, a
   variety of `ascii table formats
   <http://docs.astropy.org/en/latest/io/ascii/index.html>`_, and VO
   tables.
 * `scikit-image <http://scikit-image.org/>`_ to read popular image
   formats like ``.jpeg`` and ``.tiff``
 * `h5py <http://www.h5py.org/docs/>`_ to read HDF5 files

If Glue's predefined data loaders don't fit your needs, ou can also :ref:`write your own <custom_data_factory>` loader, and plug it into Glue.


How do I overplot catalogs on images in Glue?
---------------------------------------------
Take a look at this video. For more details, consult the :ref:`tutorial <getting_started>`.

.. raw:: html

    <center>
    <iframe src="http://player.vimeo.com/video/54940097?badge=0" width="500" height="305" frameborder="0" webkitAllowFullScreen mozallowfullscreen allowFullScreen></iframe>
    </center>

How do I use Glue with the IPython notebook?
--------------------------------------------

For IPython version >= 1.0, Glue can be started from the IPython HTML
notebook without blocking, so that both the notebook and the Glue UI
can run at once. This requires starting the notebook such that it
cooperates with Qt. To do this, start the notebook with::

    ipython notebook --gui qt

And then starting glue (probably by calling :ref:`qglue <qglue>`) from the notebook.

If you encounter an error like this::

    ValueError: API 'QString' has already been set to version 1

set the ``QT_API`` environment variable to either ``pyqt`` or ``pyside``, depending on which version of Qt you have installed::

    QT_API=pyside notebook --gui qt


Something is broken, or confusing. What should I do?
----------------------------------------------------
If you think you've found a bug in Glue, feel free to add
an issue to the `GitHub issues page <https://github.com/glue-viz/glue/issues?state=open>`_. If you have general questions, send us an `email <mailto:glue.viz@gmail.com>`_.

I have some other question...?
------------------------------
Send us an `email <mailto:glue.viz@gmail.com>`_.
