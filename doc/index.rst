
Glue Documentation
==================

.. figure:: ../glue/logo.png
   :align: center

Glue is a Python library to explore relationships within and among related datasets. Its main features include:

* **Linked Statistical Graphics.** With Glue, users can create scatter plots, histograms and images (2D and 3D) of their data. Glue is focused on the brushing and linking paradigm, where selections in any graph propagate to all others.
* **Flexible linking across data.** Glue uses the logical links that exist between different data sets to overlay visualizations of different data, and to propagate selections across data sets. These links are specified by the user, and are arbitrarily flexible.
* **Full scripting capability.** Glue is written in Python, and built on top of its standard scientific libraries (i.e., Numpy, Matplotlib, Scipy). Users can easily integrate their own python code for data input, cleaning, and analysis.

.. raw:: html

 <center>
 <iframe src="http://player.vimeo.com/video/53378575?badge=0"  width="500" height="275" frameborder="0" webkitAllowFullScreen mozallowfullscreen allowFullScreen></iframe>
 </center>

For more demos, check out the :ref:`videos <demo_videos>` page.

The latest version of glue is v0.5 - find out :ref:`whatsnew_05`.


Using glue
----------

.. toctree::
   :maxdepth: 2

   installation.rst
   getting_started/index.rst
   gui_guide/index.rst
   python_guide/index.rst
   videos.rst

Customizing glue
----------------

.. toctree::
   :maxdepth: 1

   customizing_guide/configuration.rst
   customizing_guide/customization.rst
   customizing_guide/custom_viewer.rst
   customizing_guide/full_custom_qt_viewer.rst

Getting help
------------

.. toctree::
   :maxdepth: 2

   faq.rst
   help.rst

Developer guide
---------------

.. toctree::
   :maxdepth: 1

   developer_guide/architecture.rst
   developer_guide/api.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
