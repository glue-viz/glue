.. _available_plugins:

List of available plugins
=========================

This page lists available plugin packages, as well as information on installing
these. If you are interested in writing your own plugin package, see
:ref:`writing_plugin`.

General plugins
---------------

glue-vispy-viewers: 3d viewers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **glue-vispy-viewers** plugin package adds a 3D scatter plot viewer and a 3D
volume rendering viewer to glue. This plugin package is installed by default
with glue (provided that you installed the **glueviz** package with pip or
conda). You can read up more about the functionality avalable in this plugin
in :ref:`3d-viewers`. You can check that you have this plugin installed by going
to the **Canvas** menu in glue and selecting **New Data Viewer**, or
alternatively by dragging a dataset onto the canvas area. If the 3D viewers
plugin is installed, you should see the 3D viewers in the list:

.. image:: images/3d_viewers_select.png
   :align: center
   :width: 339

If you don't see these in the list, then, you can install the 3D viewers plugin
using::

    conda install -c glueviz glue-vispy-viewers

or if you don't use conda::

    pip install glue-vispy-viewers

If you run into issues or have requests related to this plugin, or if you would
like to contribute to the development, the GitHub repository for this plugin is
at https://github.com/glue-viz/glue-vispy-viewers.

glue-jupyter: Jupyter notebook/lab viewers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We are currently developing data viewers for the Jupyter notebook and Jupyter
lab - this is not quite ready yet for general use, but if you are interested
in following on or helping with the development, the GitHub repository is at
https://github.com/glue-viz/glue-jupyter.

Plugins for Astronomy
---------------------

glue-wwt: WorldWide Telescope viewer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **glue-wwt** adds a data viewer to glue that allows users to overplot data
onto maps of the sky, powered by `WorldWide Telescope
<http://worldwidetelescope.org/>`_. You can install this plugin with::

    conda install -c glueviz glue-wwt

or if you don't use conda::

    pip install glue-wwt

Once the plugin is installed, you should see a new viewer named
**WorldWideTelescope (WWT)** in the list of available viewers when dragging a
dataset onto the main canvas in the glue application. Once you have added a
dataset to the viewer, you can select in the viewer options the columns that
give the Right Ascension and Declination of the data points (we will add support
for other coordinate systems in future). At the moment, only tables can be
shown using markers in WWT (and not images) and we don't recommend adding large
sets of points at this time (due to limitations in the way WWT deals with
annotations).

If you run into issues or have requests related to this plugin, or if you would
like to contribute to the development, the GitHub repository for this plugin is
at https://github.com/glue-viz/glue-wwt.

glue-aladin: Aladin Lite viewer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A data viewer similar to glue-wwt but for Aladin Lite
`<http://aladin.u-strasbg.fr/AladinLite/>`_, is being developed and is not quite
ready yet for general use, but if you are interested in following on or helping
with the development, the GitHub repository is at
https://github.com/glue-viz/glue-aladin.

glue-samp: Communicating with SAMP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A few common applications in astronomy support communicating via the Simple
Application Messaging Protocol (SAMP) - these include for example `DS9
<http://ds9.si.edu/site/Home.html>`_, `TOPCAT
<http://www.star.bris.ac.uk/~mbt/topcat/>`_, and `Aladin
<https://aladin.u-strasbg.fr/>`_. The **glue-samp** plugin adds the ability to
use SAMP from glue.  You can install this plugin with::

    conda install -c glueviz glue-samp

or if you don't use conda::

    pip install glue-samp

Once the plugin is installed, you can go to the **Plugins** menu and select
**Open SAMP plugin**:

.. image:: images/samp_open.png
   :align: center
   :width: 300px

A window will then appear:

.. image:: images/samp_window.png
   :align: center
   :width: 600px

Click on **Start SAMP**, and the status should change to something like
**Connected to SAMP Hub**. If you open another SAMP-enabled application such as
TOPCAT, you should now be able to send data from/to glue. To send data from glue
to another application, you can right-click (control-click on Mac) on a dataset
or subset in the glue data collection, then go to **SAMP**, then e.g. **Send to
topcat**:

.. image:: images/samp_contextual.png
   :align: center
   :width: 600px

This can be done for tables or images, and both for the main datasets and
subsets. However, note that not all SAMP-enabled application are able to
understand all types of SAMP messages. For example, while you can send images to
DS9, you will not be able to send them to TOPCAT. Conversely, DS9 may not
understand the concept of a subset.

You can also send data from other applications to glue - for more information on
doing this, see the guide for the relevant application you want to use - glue
understands messages adding images and tables, as well as messages related to
subsets.

Specviz
^^^^^^^

The `specviz <https://github.com/spacetelescope/specviz>`_ package is a
standalone application for spectral visualization and analysis, but it
incorporates a plugin for glue that makes it possible to view spectral and/or
spectral cubes open in glue. Full installation instructions are available in
the `specviz documnetation
<https://specviz.readthedocs.io/en/latest/installation.html>`__, but you can
also install specviz using::

    conda install -c glueviz specviz

or if you don't use conda::

    pip install specviz

Once specviz is installed, a new data viewer called **Specviz** will be
available, and should allow you to view spectral cubes and their subsets
as collapsed 1D spectra. More information about specviz can be found in the
`documentation <https://specviz.readthedocs.io/en/latest/index.html>`__, as well
as at the `GitHub repository <https://github.com/spacetelescope/specviz>`_.

CubeViz and MOSViz
^^^^^^^^^^^^^^^^^^

**CubeViz** and **MOSViz** are applications developed at the Space Science
Institute and built on top of glue for the visualization of IFU Spectral Cubes
and for Multi-Object Spectroscopy (MOS) respectively. To find out more about
using these, see https://cubeviz.readthedocs.io and
https://mosviz.readthedocs.io. As for other packages mentioned on this page,
you can easily install these using::

    conda install -c glueviz cubeviz mosviz

or if you don't use conda::

    pip install cubeviz mosviz

Plugins for Medicine
--------------------

glue-medical

Plugins for Geosciences
-----------------------

glue-geospatial
