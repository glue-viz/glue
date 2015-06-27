Writing a fully customized Qt viewer (advanced)
===============================================

Motivation
----------

The :func:`~glue.custom_viewer` function and the
:class:`~glue.qt.custom_viewer.CustomViewer` class described in
:doc:`custom_viewer` are well-suited to developing new custom viewers that
include some kind of Matplotlib plot. But in some cases, you may want to
write a Qt data viewer that doesn't depend on Matplotlib, or may use an
existing widget. In this tutorial, we will assume that you have implemented a
Qt widget that contains the functionality you want, and we will focus on
looking at how to get it to work inside glue.

If you don't already have an existing widget, but want to make sure it will
work outside glue, start off by developing the widget outside of glue, then
use the instructions below to make it usable inside glue.

Displaying the widget in glue
-----------------------------

Let's imagine that you have a Qt widget class called ``MyWidget`` the
inherits from ``QWidget`` and implements a specific type of visualization you
are interested in::

    class MyWidget(QWidget):
        ...

Now let's say we want to use this widget in glue, without having to change
anything in ``MyWidget``. The best way to do this is to create a new class,
``MyGlueWidget``, that will wrap around ``MyWidget`` and make it
glue-compatible. The glue widget should inherit from
:class:`~glue.qt.widgets.data_viewer.DataViewer` (this class does a few
boilerplate things such as, for example, adding the ability to drag and drop
data onto your data viewer).

The simplest glue widget wrapper that you can write that will show
``MyWidget`` is::

    from glue.qt.widgets.data_viewer import DataViewer

    class MyGlueWidget(DataViewer):

        def __init__(self, session, parent=None):
            super(MyGlueWidget, self).__init__(session, parent=parent)
            self.my_widget = MyWidget()
            self.setCentralWidget(self.my_widget)

    # Register the viewer with glue
    from glue.config import qt_client
    qt_client.add(MyGlueWidget)

If you put the contents above into a ``config.py`` file then launch glue in
the same folder as the ``config.py`` file, you will then be able to go to the
**Canvas** menu, select **New Data Viewer**, and you should then be presented
with the window to select a data view, which should contain an 'Override
This' entry:

.. image:: images/select_override.png
   :width: 200px
   :align: center

To give your viewer a more meaningful name, you should give your class an
attribute called ``LABEL``::

    class MyGlueWidget(DataViewer):

        LABEL = "My first data viewer"

        def __init__(self, session, parent=None):
            super(MyGlueWidget, self).__init__(session, parent=parent)
            self.my_widget = MyWidget()
            self.setCentralWidget(self.my_widget)

Passing data to the widget
--------------------------

Now we want to be able to pass data to this viewer. To do this, you should
define the ``add_data`` method which should take a single argument and return
`True` if adding the data succeeded, and `False` otherwise. So for now, let's
simply return True and do nothing::

        def add_data(self, data):
            return True

Now you can open glue again, and this time you should be able to load a
dataset the usual way. When you drag this dataset onto the main canvas area,
you will be able to then select your custom viewer, and it should appear
(though the data itself will not). You can now expand the ``add_data`` method
to actually add the data to ``MyWidget``, by accessing ``self.my_widget``,
for example::

        def add_data(self, data):
            self.my_widget.plot(data)
            return True

However, this will simply plot the initial data and plot more data if you
drag datasets onto the window, but you will not for example be able to remove
datasets, show subsets, and so on. In some cases, that may be fine, and you
can stop at this point, but in other cases, if you want to define a way to
interact with subsets, propagate selections, and so on, you will need to set
up a glue client, which is discussed in `Setting up a client`_. But first, let's take a look at how we can add side panels in the dashboard which can include for example options for controlling the appearance or contents of your visualization.

Adding side panels
------------------

In the glue interface, under the data manager is an area we refer to as the
dashboard, where different data viewers can include options for controlling
the appearance or content of visualizations (this is the area indicated as C
in :doc:getting-started). You can add any widget to the two available spaces.

In your wrapper class, ``MyGlueWidget`` in the example above, you will need
to define two methods called ``layer_view`` and ``options_widget``, which
each return an instantiated widget that should be included in the dashboard.
These names are because the top space is usually used for showing which
layers are included in a plot, and the bottom space is used for options (such
as the number of bins in histograms).

For example, you could do::

    class MyGlueWidget(DataViewer):

        ...

        def __init__(self, session, parent=None):
            ...
            self._layer_view = UsefulWidget(...)
            self._options_widget = AnotherWidget(...)

        ...

        def layer_view(self):
            return self._layer_view

        def options_widget(self):
            return self._options_widget

Note that despite the name, you can actually set the widgets to what you
want, and the important thing is that ``layer_view`` is the top one, and
``options_widget`` the bottom one.

Setting up a client
-------------------

Once the data viewer has been instantiated, the main glue application will call the ``register_to_hub`` method on the data viewer, and will pass it the hub as an argument. This allows you to set up your data viewer as a client that can listen to specific messages from the hub::

    from glue.core.message import DataCollectionAddMessage

    class MyGlueWidget(DataViewer):

        ...

        def register_to_hub(self, hub):

            super(MyGlueWidget, self).register_to_hub(hub)

            # Now we can subscribe to messages with the hub

            hub.subscribe(self,
                          DataUpdateMessage,
                          handler=self._update_data)

        def _update_data(self, msg):

            # Process DataUpdateMessage here
