.. _programmatic:

====================================
Programmatically configuring viewers
====================================

Viewers in Glue are designed to be easily configured with Python. As much as
possible, viewer settings are controlled by simple properties on the ``state``
attribute of data viewer objects. For example::

    import numpy as np

    from glue.core import Data, DataCollection
    from glue.app.qt.application import GlueApplication
    from glue.viewers.scatter.qt import ScatterViewer

    # create some data
    d = Data(x=np.random.random(100), y=np.random.random(100))
    dc = DataCollection([d])

    # create a GUI session
    ga = GlueApplication(dc)

    # plot x vs y, flip the x axis, log-scale y axis
    scatter = ga.new_data_viewer(ScatterViewer)
    scatter.add_data(d)

    # Modify viewer-level options
    scatter.state.x_att = d.id['x']
    scatter.state.y_att = d.id['y']
    scatter.state.y_log = True

    # Modify settings for the (only) layer shown
    scatter.state.layers[0].color = 'blue'

    # show the GUI
    ga.start()

Viewer Options
==============

The ``state`` attribute for each viewer is an instance of a viewer state class.
Each viewer state object then has a ``layers`` attribute that can be used to
control individual layers in the viewer (as shown above).

The following table lists for each built-in viewer the classes defining the state
for each viewer/layer type. By clicking on the name of the class, you will access
a page from the API documentation which will list the available attributes.

=================== ========================= ======================= ========================
Viewer              Viewer state              Data layer state        Subset layer state
=================== ========================= ======================= ========================
|scatter_viewer|    |scatter_viewer_state|    |scatter_layer_state|   |scatter_layer_state|
|image_viewer|      |image_viewer_state|      |image_data_state|      |image_subset_state|
|histogram_viewer|  |histogram_viewer_state|  |histogram_layer_state| |histogram_layer_state|
|profile_viewer|    |profile_viewer_state|    |profile_layer_state|   |profile_layer_state|
=================== ========================= ======================= ========================

.. |scatter_viewer| replace:: :class:`~glue.viewers.scatter.qt.ScatterViewer`
.. |scatter_viewer_state| replace:: :class:`~glue.viewers.scatter.state.ScatterViewerState`
.. |scatter_layer_state| replace:: :class:`~glue.viewers.scatter.state.ScatterLayerState`

.. |image_viewer| replace:: :class:`~glue.viewers.image.qt.ImageViewer`
.. |image_viewer_state| replace:: :class:`~glue.viewers.image.state.ImageViewerState`
.. |image_data_state| replace:: :class:`~glue.viewers.image.state.ImageLayerState`
.. |image_subset_state| replace:: :class:`~glue.viewers.image.state.ImageSubsetLayerState`

.. |histogram_viewer| replace:: :class:`~glue.viewers.histogram.qt.HistogramViewer`
.. |histogram_viewer_state| replace:: :class:`~glue.viewers.histogram.state.HistogramViewerState`
.. |histogram_layer_state| replace:: :class:`~glue.viewers.histogram.state.HistogramLayerState`

.. |profile_viewer| replace:: :class:`~glue.viewers.profile.qt.ProfileViewer`
.. |profile_viewer_state| replace:: :class:`~glue.viewers.profile.state.ProfileViewerState`
.. |profile_layer_state| replace:: :class:`~glue.viewers.profile.state.ProfileLayerState`

Customizing Plots with Matplotlib
=================================

If you want, you can directly manipulate the Matplotlib plot objects that
underlie Glue viewers. This can be useful if you want to create static plots with
custom annotation, styles, etc.

From the GUI
------------
Open the IPython terminal window. The ``application.viewers`` variable
is a list of lists of all the
open viewer windows. Each inner list contains the data viewers
open on a single tab. Every viewer has an ``axes`` attribute,
which points to a :class:`Matplotlib Axes <matplotlib.axes.Axes>`
object::

    viewer = application.viewers[0][0]
    ax = viewer.axes
    ax.set_title('Custom title')
    ax.figure.canvas.draw()  # update the plot

From a script
-------------

Save the current glue session via ``File->Save Session``. You can
reload this session programmatically as follows::

    from glue.app.qt.application import GlueApplication
    app = GlueApplication.restore('output.glu', show=False)
    viewer = app.viewers[0][0]
    ax = viewer.axes
