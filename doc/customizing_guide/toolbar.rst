Toolbars
========

Writing a custom button for a viewer toolbar
--------------------------------------------

To create a tool to include in a viewer (either one of the built-in viewers or a
custom viewer), you will need to write a short class. There are two types of
toolbar buttons: ones that can be checked and unchecked, and ones that simply
trigger an event when pressed, but do not remain pressed. These are described
in the following sub-sections.

Non-checkable buttons
^^^^^^^^^^^^^^^^^^^^^

The basic structure for a non-checkable button is:

.. code:: python

    from glue.config import toolbar_mode
    from glue.viewers.common.qt.tool import Tool

    @toolbar_mode
    class MyCustomButton(Tool):

        icon = 'myicon.png'
        tool_id = 'custom_mode'
        action_text = 'Does cool stuff'
        tool_tip = 'Does cool stuff'
        shortcut = 'D'

        def __init__(self, viewer):
            super(MyCustomMode, self).__init__(viewer)

        def activate(self):
            pass

        def close(self):
            pass

The class-level variables set at the start of the class are as follows:

* ``icon``: this should be set either to the name of a built-in glue icon, or
  to the path to a PNG file to be used for the icon. Note that this should
  **not** be a ``QIcon`` object.

* ``tool_id``: a unique string that identifies this tool. If you create a
  button/mode that has the same ``tool_id`` as an existing tool already
  implemented in glue, you will overwrite the existing tool. This is never shown
  to the user.

* ``action_text``: a string describing the tool. This is shown in the status bar
* at the bottom of the viewer whenever the button is active.

* ``tool_tip``: this should be a string that will be shown when the user hovers
  above the button in the toolbar. This can include instructions on how to use
  the button.

* ``shortcut``: this should be a key that the user can press when the viewer is
  active, which will activate the button. This can include modifier keys, e.g.
  ``Ctrl+A`` or ``Ctrl+Shift+U``, but can also just be a single key, e.g. ``K``.

When the user presses the button, the ``activate`` method is called. In this
method, you can write any code including code that may for example open a Qt
window, or change the state of the viewer (for example changing the zoom or
field of view). You can access the viewer instance with ``self.viewer``.
Finally, when the viewer is closed the ``close`` method is called, so you should
use this to do any necessary cleanup.

Checkable buttons
^^^^^^^^^^^^^^^^^

The basic structure for a checkable button is similar to non-checkable buttons,
but with an additional ``deactivate`` method:

.. code:: python

    from glue.config import toolbar_mode
    from glue.viewers.common.qt.tool import CheckableTool

    @toolbar_mode
    class MyCustomButton(CheckableTool):

        icon = 'myicon.png'
        tool_id = 'custom_mode'
        action_text = 'Does cool stuff'
        tool_tip = 'Does cool stuff'
        shortcut = 'D'

        def __init__(self, viewer):
            super(MyCustomMode, self).__init__(viewer)

        def activate(self):
            pass

        def deactivate(self):
            pass

        def close(self):
            pass

When the button is checked, the ``activate`` method is called, and when the
button is unchecked (either by clicking on it again, or if the user clicks on
another button), the ``deactivate`` method is called. As before, when the viewer
is closed, the ``close`` method is called.

Button menus
^^^^^^^^^^^^

For both checkable and non-checkable buttons, it is possible to show a menu
when the user clicks on the button. To do this, simply add a ``menu_actions``
method to your class:

.. code:: python

    def menu_actions(self):
        return []

This method should return a list of ``QActions`` which can include e.g. icons,
text, and callbacks.

.. note:: In future, we will allow this to be done in a way that
          does not rely on Qt QActions.

Customizing the content of a toolbar
------------------------------------

When defining a button as above, the ``@toolbar_mode`` decorator ensures that
the mode is registered with glue, but does not add it to any specific viewer.
Which buttons are shown for a viewer is controlled by the ``modes`` class-level
attribute on viewers:

.. code:: python

    >>> from glue.viewers.image.qt import ImageWidget
    >>> ImageWidget.tools
    ['Rectangle', 'X range', 'Y range', 'Circle', 'Polygon', 'COLORMAP']

The strings in the ``modes`` list correspond to the ``tool_id`` attribute on the
button/mode classes. If you want to add an existing or custom button to a
viewer, you can therefore simply do e.g.:

.. code:: python

    from glue.viewers.image.qt import ImageWidget
    ImageWidget.tools.append('custom_mode')

Including toolbars in custom viewers
------------------------------------

When defining a data viewer (as described in :doc:`full_custom_qt_viewer`), it
is straightforward to add a toolbar that can then be used to add buttons. To do
this, when defining your `glue.viewers.common.qt.data_viewer.DataViewer` subclass,
you should also specify the ``_toolbar_cls`` and ``modes`` class-level
attributes, which should give the class to use for the toolbar, and the default
modes that should be present in the toolbar:

.. code:: python

    from glue.viewers.common.qt.data_viewer import DataViewer
    from glue.viewers.common.qt.toolbar import BasicToolbar

    class MyViewer(DataViewer):

        _toolbar_cls = BasicToolbar
        tools = []

In the example above, the viewer will include an empty toolbar. There are
currently two main classes available for toolbars:

* :class:`~glue.viewers.common.qt.toolbar.BasicToolbar`: this is the most basic
  kind of toolbar - it comes with no buttons by default.

* :class:`~glue.viewers.common.qt.mpl_toolbar.MatplotlibViewerToolbar`: this is a
  subclass of :class:`~glue.viewers.common.qt.toolbar.BasicToolbar` that includes
  the standard Matplotlib buttons by default (home, zoom, pan, etc.). This
  toolbar can only be used if your data viewer includes a Matplotlib canvas
  accessible at ``viewer.canvas``.
