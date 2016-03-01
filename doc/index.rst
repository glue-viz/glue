
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

**The latest version of glue is v0.6** - see our :ref:`overview of changes in 0.6 <whatsnew_06>`

Getting started
---------------

Glue is designed with "data-hacking" workflows in mind, and can be used in
different ways. For instance, you can simply make use of the graphical Glue
application as is, and never type a line of code. However, you can also
interact with Glue via Python in different ways:

* Using the IPython terminal built-in to the Glue application
* Sending data in the form of NumPy arrays or Pandas DataFrames
  to Glue for exploration from a Python or IPython session.
* Customizing/hacking your Glue setup using ``config.py`` files, including
  automatically loading and clean data before starting Glue, writing custom
  functions to parse files in your favorite file format, writing custom
  functions to link datasets, or creating your own data viewers.

Glue thus blurs the boundary between GUI-centric and code-centric data
exploration. In addition, it is also possible to develop your own plugin
packages for Glue that you can distribute to users, and you can also make use
of the Glue framework in your own application to provide data linking
capabilities.

In the following sections, we cover the different ways of using Glue from the
Glue application to the more advanced ways of interacting with Glue from Python.

For instructions on installing Glue, head over to :doc:`installation`.
 
Using the Glue application
--------------------------

.. toctree::
   :maxdepth: 2
   
   getting_started/index.rst
   gui_guide/index.rst

Interacting with data from Python
---------------------------------

.. toctree::
   :maxdepth: 1

   python_guide/ipython_terminal.rst
   python_guide/data_tutorial.rst
   python_guide/glue_from_python.rst

Customizing/Hacking Glue
------------------------

.. toctree::
   :maxdepth: 1

   python_guide/data_viewer_options.rst

   customizing_guide/configuration.rst
   customizing_guide/customization.rst
   customizing_guide/custom_viewer.rst
   customizing_guide/full_custom_qt_viewer.rst

   python_guide/liveupdate.rst

Getting help
------------

.. toctree::
   :maxdepth: 1
   
   videos.rst
   faq.rst
   help.rst
   
Developer guide
---------------

.. toctree::
   :maxdepth: 1

   developer_guide/developer_guide.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
