How to write a fully customized Qt viewer
=========================================

Motivation
----------

The :func:`custom data viewer <glue.custom_viewer>` function and the
:class:`~glue.qt.custom_viewer.CustomViewer` class described in
:doc:`custom_viewer` are well-suited to developing new custom viewers that include some kind of Matplotlib plot. But in some cases, you may want to write a Qt data viewer that doesn't depend on Matplotlib, or may use an existing widget.

First steps
-----------

Let's imagine that you have a Qt widget class called ``MyWidget`` the
inherits from ``QWidget`` and implements a specific type of visualization you
are interested in::

    class MyWidget(QWidget):
        ...

Now let's say we want to use this widget in glue, without having to change anything in ``MyWidget``. The best way to do this is to create a new class, ``MyGlueWidge``, that will wrap around ``MyWidget`` and make it glue-compatible. The glue widget should inherit from :class:`~glue.qt.widgets.data_viewer.DataViewer` (this class does a few boilerplate things such as for example adding the ability to drag and drop data onto your data viewer.

The simplest glue widget wrapper that you can write that will show
``MyWidget`` is::

    from glue.qt.widgets.data_viewer import DataViewer

    class MyGlueWidget(DataViewer):

        def __init__(self, session=None, parent=None):
            super(MyGlueWidget, self).__init__(session=session, parent=parent)
            self.my_widget = MyWidget()
            self.setCentralWidget(self.my_widget)

    # Register the viewer with glue
    from glue.config import qt_client
    qt_client.add(MyGlueWidget)

If you put the contents above into a ``config.py`` file then launch glue in the same folder as the ``config.py`` file, you will then be able to go to the **Canvas** menu, select **New Data Viewer**, and you should then be presented with the window to select a data view, which should contain an 'Override This' entry:

.. image:: images/select_override.png

To give your viewer a more meaningful name, you should give it an attribute called ``LABEL`` containing a string that will be used as the data viewer name::

    class MyGlueWidget(DataViewer):

        LABEL = "My first data viewer"

        def __init__(self, session=None, parent=None):
            super(MyGlueWidget, self).__init__(session=session, parent=parent)
            self.my_widget = MyWidget()
            self.setCentralWidget(self.my_widget)

Now we want to be able to pass data to this viewer. To do this, you should define the ``add_data`` method which should take a single argument and return `True` if adding the data succeeded, and `False` otherwise. So for now, let's simply return True and do nothing::

        def add_data(self, data):
            return True

Now you can open glue again, and this time you should be able to load a dataset the usual way. When you drag this dataset onto the main canvas area, you will be able to then select your custom viewer, and it should appear (though the data itself will not). You can now expand the ``add_data`` method to actually add the data to ``MyWidget``, by accessing ``self.my_widget``, for example::

    def add_data(self, data):
        self.my_widget.plot(data)
        return True

However, this will simply plot the initial data and plot more data if you drag datasets onto the window, but you will not for example be able to remove datasets, show subsets, and so on. In some cases, that may be fine, and you can stop at this point, but in other cases, if you want to define a way to interact with subsets, propagate selections, and so on, you will need to set up a glue client.

Setting up a client
-------------------