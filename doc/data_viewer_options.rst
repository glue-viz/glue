==================================
Programmatically configuring plots
==================================

Data Viewers are designed to be easily configured
from python scripts. For example::

    from glue.core import Data, DataCollection
    from glue.qt import GlueApplication
    from glue.qt.widgets import ScatterWidget
    import numpy as np

    # create some data
    d = Data(x=np.random.random(100), y=np.random.random(100))
    dc = DataCollection([d])

    # create a GUI session
    ga = GlueApplication(dc)

    # plot x vs y, flip the x axis, log-scale y axis
    scatter = ga.new_data_viewer(ScatterWidget)
    scatter.add_data(d)
    scatter.xatt = d.id['x']
    scatter.yatt = d.id['y']
    scatter.xflip = True
    scatter.ylog = True

    # show the GUI
    ga.start()

Here are the settings associated with each data viewer:

.. currentmodule:: glue.qt.widgets.scatter_widget

:class:`Scatter Plots <ScatterWidget>`
--------------------------------------

.. autosummary::
    ~ScatterWidget.xlog
    ~ScatterWidget.ylog
    ~ScatterWidget.xflip
    ~ScatterWidget.yflip
    ~ScatterWidget.xmin
    ~ScatterWidget.xmax
    ~ScatterWidget.ymin
    ~ScatterWidget.ymax
    ~ScatterWidget.hidden
    ~ScatterWidget.xatt
    ~ScatterWidget.yatt

.. currentmodule:: glue.qt.widgets.image_widget

:class:`Image Viewer <ImageWidget>`
------------------------------------

.. autosummary::
    ~ImageWidget.data
    ~ImageWidget.attribute
    ~ImageWidget.rgb_mode
    ~ImageWidget.slice

.. currentmodule:: glue.qt.widgets.histogram_widget

:class:`Histogram Viewer <HistogramWidget>`
---------------------------------------------

.. autosummary::

    ~HistogramWidget.xmin
    ~HistogramWidget.xmax
    ~HistogramWidget.normed
    ~HistogramWidget.autoscale
    ~HistogramWidget.cumulative
    ~HistogramWidget.nbins
    ~HistogramWidget.xlog
    ~HistogramWidget.ylog