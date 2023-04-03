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
 <iframe width="640" height="390" src="https://www.youtube.com/embed/qO3RQiRjWA4?rel=0" frameborder="0" allowfullscreen></iframe>
 </center>

For more demos, check out the :ref:`videos <demo_videos>` page. You can also
`follow us on Twitter <https://twitter.com/glueviz>`_ for previews of upcoming
functionality!

**The latest release of glue is v1.0.x** - find out :ref:`what's new in glue! <whatsnew>`

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
packages for Glue that you can distribute to users separately, and you can also
make use of the Glue framework in your own application to provide data linking
capabilities.

To see how glue compares with other open-source and commercial data visualization
solutions, you can view  `this comparison table <https://docs.google.com/spreadsheets/d/1u_sgRUtnqEA067C5ffsMRkUpb0XUAMIoq6QfqivHG78/pubhtml?gid=0&single=true>`_.

In the following sections, we cover the different ways of using Glue from the
Glue application to the more advanced ways of interacting with Glue from Python.

.. note:: For any questions or help with using glue, you can always join the
          `user support mailing list <https://groups.google.com/forum/#!forum/glue-viz>`_
          or ask questions on `Slack <https://glueviz.slack.com>`_
          (note that you will need to sign up for an account `here <https://join.slack.com/t/glueviz/shared_invite/zt-1p5amn7ee-wTZCJJNlgxCTRup3SrSh3g>`_).

Using the Glue application
--------------------------

.. toctree::
   :maxdepth: 2

   installation/installation.rst
   getting_started/index.rst
   gui_guide/index.rst

Interacting with data from Python
---------------------------------

.. toctree::
   :maxdepth: 1

   python_guide/ipython_terminal.rst
   python_guide/data_tutorial.rst
   python_guide/glue_from_python.rst
   python_guide/data_translation.rst

Domain-specific guides
----------------------

.. toctree::
   :maxdepth: 1

   gui_guide/dendro.rst

Customizing/Hacking Glue
------------------------

.. toctree::
   :maxdepth: 1

   customizing_guide/introduction.rst
   customizing_guide/available_plugins.rst
   customizing_guide/configuration.rst
   customizing_guide/customization.rst
   customizing_guide/writing_plugin.rst
   customizing_guide/coordinates.rst
   python_guide/data_viewer_options.rst
   customizing_guide/custom_viewer.rst
   python_guide/liveupdate.rst
   customizing_guide/fitting.rst
   customizing_guide/units.rst

Advanced customization
----------------------

.. toctree::
   :maxdepth: 1

   customizing_guide/viewer.rst
   customizing_guide/qt_viewer.rst
   customizing_guide/matplotlib_qt_viewer.rst
   customizing_guide/toolbar.rst
   developer_guide/data.rst

Getting help
------------

.. toctree::
   :maxdepth: 1

   videos.rst
   faq.rst
   help.rst
   known_issues.rst

.. _architecture:

The Glue architecture
---------------------

The pages below take you through the main infrastructure in Glue, and in
particular how selections, linking, and communications are handled internally.
You don't need to understand all of this in order to get started with
contributing, but in order to tackle some of the more in-depth issues, this
will become important. This is not meant to be a completely exhaustive guide,
but if there are areas that you feel could be explained better, or are missing
and would be useful, please let us know!

.. toctree::
   :maxdepth: 1

   developer_guide/selection.rst
   developer_guide/communication.rst
   developer_guide/linking.rst

Information on the Data framework is available in :ref:`data_tutorial` and is not repeated here.

.. _devdocs:

Developing Glue
---------------

.. toctree::
   :maxdepth: 2

   developer_guide/developer_guide.rst

Acknowledging glue
------------------

If you use glue for research presented in a publication, please consider citing
the following two references:

* `Beaumont et al. (2015), Hackable User Interfaces In Astronomy with
  Glue <http://adsabs.harvard.edu/abs/2015ASPC..495..101B>`_

* `Robitaille et al (2017) glueviz v0.13.1: multidimensional data exploration
  <https://zenodo.org/record/1237692>`_

The first is a conference proceedings describing glue, while the second is the
software itself.

Publications
------------

* `Goodman et al. (2012), Principles of high-dimensional data
  visualization in astronomy <http://adsabs.harvard.edu/abs/2012AN....333..505G>`_
* `Beaumont et al. (2015), Hackable User Interfaces In Astronomy with
  Glue <http://adsabs.harvard.edu/abs/2015ASPC..495..101B>`_
* `Robitaille et al (2017) glueviz v0.10: multidimensional data exploration
  <https://zenodo.org/record/293197>`_

API
---

.. toctree::
   :maxdepth: 1

   developer_guide/api.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
