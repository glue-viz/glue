.. _custom-toolbars:

Custom tools for viewers and custom toolbars
============================================

Writing a custom tool for a viewer toolbar
------------------------------------------

Here we take a look at how to create a tool to include in a viewer's toolbar
(either one of the built-in viewers or a custom viewer) There are two types of
tools: ones that can be checked and unchecked, and ones that simply trigger an
event when pressed, but do not remain pressed. These are described in the
following two sub-sections.

Non-checkable tools
^^^^^^^^^^^^^^^^^^^

The basic structure for a non-checkable tool is:

.. code:: python

    from glue.config import viewer_tool
    from glue.viewers.common.tool import Tool

    @viewer_tool
    class MyCustomTool(Tool):

        icon = 'myicon.png'
        tool_id = 'custom_tool'
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
  tool that has the same ``tool_id`` as an existing tool already implemented in
  glue, you will get an error.

* ``action_text``: a string describing the tool. This is not currently used,
  but would be the text that would appear if the tool was accessible by a menu.

* ``tool_tip``: this should be a string that will be shown when the user hovers
  above the button in the toolbar. This can include instructions on how to use
  the tool.

* ``shortcut``: this should be a string giving a key that the user can press
  when the viewer is active, which will activate the tool. This can include
  modifier keys, e.g. ``'Ctrl+A'`` or ``'Ctrl+Shift+U'``, but can also just be
  a single key, e.g. ``'K'``. If present, the shortcut is added at the end of
  the tooltip. If multiple tools in a viewer have the same shortcut, a warning
  will be emitted, and only the first tool registered with a particular
  shortcut will be accessible with that shortcut.

When the user presses the tool icon, the ``activate`` method is called. In this
method, you can write any code including code that may for example open a Qt
window, or change the state of the viewer (for example changing the zoom or
field of view). You can access the viewer instance with ``self.viewer``.
Finally, when the viewer is closed the ``close`` method is called, so you should
use this to do any necessary cleanup.

The ``@viewer_tool`` decorator tells glue that this class represents a viewer
tool, and you will then be able to add the tool to any viewers (see
:ref:`toolbar_content`) using the ``tool_id``.

Checkable tools
^^^^^^^^^^^^^^^

The basic structure for a checkable tool is similar to the above, but with an
additional ``deactivate`` method, and a ``status_tip`` attribute:

.. code:: python

    from glue.config import viewer_tool
    from glue.viewers.common.tool import CheckableTool

    @viewer_tool
    class MyCustomButton(CheckableTool):

        icon = 'myicon.png'
        tool_id = 'custom_tool'
        action_text = 'Does cool stuff'
        tool_tip = 'Does cool stuff'
        status_tip = 'Instructions on what to do now'
        shortcut = 'D'

        def __init__(self, viewer):
            super(MyCustomMode, self).__init__(viewer)

        def activate(self):
            pass

        def deactivate(self):
            pass

        def close(self):
            pass

When the tool icon is pressed, the ``activate`` method is called, and when the
button is unchecked (either by clicking on it again, or if the user clicks on
another tool icon), the ``deactivate`` method is called. As before, when the
viewer is closed, the ``close`` method is called. The ``status_tip`` is a
message shown in the status bar of the viewer when the tool is active. This can
be used to provide instructions to the user as to what they should do next.

Drop-down menus
^^^^^^^^^^^^^^^

For both checkable and non-checkable tools, it is possible to show a menu
when the user clicks on the icon. To do this, simply add a ``menu_actions``
method to your class:

.. code:: python

    def menu_actions(self):
        return []

This method should return a list of ``QActions`` which can include e.g. icons,
text, and callbacks.

.. note:: In future, we will allow this to be done in a way that
          does not rely on Qt QActions.

.. _toolbar_content:

Customizing the content of a toolbar
------------------------------------

When defining a tool as above, the ``@viewer_tool`` decorator ensures that
the tool is registered with glue, but does not add it to any specific viewer.
Which buttons are shown for a viewer is controlled by the ``tools`` class-level
attribute on viewers:

.. code:: python

    >>> from glue.viewers.image.qt import ImageViewer
    >>> ImageViewer.tools
    ['select:rectangle', 'select:xrange', 'select:yrange',
     'select:circle', 'select:polygon', 'image:colormap']

The strings in the ``tools`` list correspond to the ``tool_id`` attribute on the
tool classes. If you want to add an existing or custom button to a viewer, you
can therefore simply do e.g.:

.. code:: python

    from glue.viewers.image.qt import ImageViewer
    ImageViewer.tools.append('custom_tool')

Including toolbars in custom viewers
------------------------------------

When defining a data viewer (as described in :ref:`state-qt-viewer`), it
is straightforward to add a toolbar that can then be used to add tools. To do
this, when defining your
:class:`~glue.viewers.common.qt.data_viewer.DataViewer` subclass,
you should also specify the ``_toolbar_cls`` and ``tools`` class-level
attributes, which should give the class to use for the toolbar, and the default
tools that should be present in the toolbar:

.. code:: python

    from glue.viewers.common.qt.data_viewer import DataViewer
    from glue.viewers.common.qt.toolbar import BasicToolbar

    class MyViewer(DataViewer):

        _toolbar_cls = BasicToolbar
        tools = ['custom_tool']

In the example above, the viewer will include an toolbar with one tool (the one
we defined above). Currently the only toolbar class that is defined
is :class:`~glue.viewers.common.qt.toolbar.BasicToolbar`.

Note that the toolbar is set up after ``__init__`` has run. Therefore, if you
want to do any custom set-up to the toolbar after it has been set up, you
should overload the ``initialize_toolbar`` method, e.g:

.. code:: python

    class MyViewer(DataViewer):

        _toolbar_cls = BasicToolbar
        tools = ['custom_tool']

        def initialize_toolbar(self):
            super(MyViewer, self).initialize_toolbar()
            # custom code here

In ``initialize_toolbar`` (and elsewhere in the class) you can then access the
tool instances using ``self.toolbar.tools`` (which is a dictionary where each
key is a ``tool_id``).

By default, tools are inherited from parent classes, but this can be controlled
using the ``inherit_tools`` class-level attribute - for example, the following
will result in only the ``custom_tool`` being available, and nothing else:

.. code:: python

    class MyImageViewer(ImageViewer):

        tools = ['custom_tool']
        inherit_tools = False

Available tools
---------------

The following tools are available by default (note that not all tools can be
used in all viewers, click on each tool class name to find out more):

======================  ========================================================
Tool ID                 Class
======================  ========================================================
``'select:circle'``     :class:`~glue.viewers.matplotlib.toolbar_mode.CircleMode`
``'select:pick'``       :class:`~glue.viewers.matplotlib.toolbar_mode.PickMode`
``'select:polygon'``    :class:`~glue.viewers.matplotlib.toolbar_mode.PolyMode`
``'select:rectangle'``  :class:`~glue.viewers.matplotlib.toolbar_mode.RectangleMode`
``'select:xrange'``     :class:`~glue.viewers.matplotlib.toolbar_mode.HRangeMode`
``'select:yange'``      :class:`~glue.viewers.matplotlib.toolbar_mode.VRangeMode`
``'image:colormap'``    :class:`~glue.viewers.matplotlib.qt.toolbar_mode.ColormapMode`
``'image:contrast'``    :class:`~glue.viewers.matplotlib.qt.toolbar_mode.ContrastMode`
======================  ========================================================
