.. _state-viewer:

Writing a custom viewer for glue
================================

Motivation
----------

The simple way of defining new custom viewers described in :doc:`custom_viewer`
are well-suited to developing new custom viewers that include simple Matplotlib
plots, and for now is limited to Qt-based viewers. But in some cases, you may
want to write a data viewer with more customized functionality, or that doesn't
depend on Matplotlib or Qt and may use an existing third-party widget.

In this tutorial, we will take a look at the pieces needed to build a data
viewer. The sections here are relevant regardless of whether you are building a
data viewer for e.g. Qt or Jupyter. If you are interested in building a Qt-based
viewer, you can then proceed to :ref:`state-qt-viewer`. If you are interested in
building a Matplotlib-based Qt viewer, you can then also make use of the
``glue.viewers.matplotlib`` sub-package to simplify things as described in
:ref:`matplotlib-qt-viewer`.

Terminology
-----------

When we talk about a *data viewer*, we mean specifically one of the
visualizations in glue (e.g. scatter plot, histogram, network diagram, etc.). Inside each
visualization, there may be multiple datasets or subsets shown. For example, a
dataset might be shown as markers of a certain color, while a subset might be
shown in a different color. We refer to these as *layers* in the visualization,
and these typically appear in a list on the left of the glue application window.

State classes
-------------

Overview
^^^^^^^^

The first piece to construct when developing a new data viewer are *state*
classes for the data viewer and layers, which you can think of as a conceptual
representation of the data viewer and layers, but doesn't contain any code
specific to e.g. Qt or Jupyter or even the visualization library you are using.
As an example, a scatter viewer will have a state class that indicates which
attributes are shown on which axes, and what the limits of the axes are. Each
layer then also has a state class which includes information for example about
what the color of the layer should be, and whether it is currently visible or
not.

Viewer state
^^^^^^^^^^^^

To create a viewer, we import the base
:class:`~glue.viewers.common.state.ViewerState` class, as well as the
:class:`~echo.CallbackProperty` class::

    from glue.viewers.common.state import ViewerState
    from echo import CallbackProperty

The latter is used to define properties on the state class and we can attach
callback functions to them (more on this soon). Let's now imagine we want to
build a simple scatter plot viewer. Our state class would then look like::

    class TutorialViewerState(ViewerState):
        x_att = CallbackProperty(docstring='The attribute to use on the x-axis')
        y_att = CallbackProperty(docstring='The attribute to use on the y-axis')

Once a state class is defined with callback properties, it is possible to
attach callback functions to them::

    >>> def on_x_att_change(value):
    ...     print('x_att has changed and is now', value)
    >>> state = TutorialViewerState()
    >>> state.add_callback('x_att', on_x_att_change)
    >>> state.x_att = 'a'
    x_att has changed and is now a

What this means is that when you are defining the state class for your viewer,
think about whether you want to change certain properties based on others. For
example we could write a state class that changes x to match y (but not y to
match x)::

  class TutorialViewerState(ViewerState):

      x_att = CallbackProperty(docstring='The attribute to use on the x-axis')
      y_att = CallbackProperty(docstring='The attribute to use on the y-axis')

      def __init__(self, *args, **kwargs):
          super(TutorialViewerState).__init__(*args, **kwargs)
          self.add_callback('y_att', self._on_y_att_change)

      def _on_y_att_change(self, value):
          self.x_att = self.y_att

The idea is to implement as much of the logic as possible here rather than
relying on e.g. Qt events, so that your class can be re-used for e.g. both a Qt
and Jupyter data viewer.

Note that the :class:`~glue.viewers.common.state.ViewerState` defines one
property by default, which is ``layers`` - a container that is used to store
:class:`~glue.viewers.common.state.LayerState` objects (see `Layer state`_).
You shouldn't need to add/remove layers from this manually, but you can attach
callback functions to ``layers`` in case any of the layers change.

Layer state
^^^^^^^^^^^

Similarly to the viewer state, you need to also define a state class for
layers in the visualization using :class:`~glue.viewers.common.state.LayerState`::

    from glue.viewers.common.state import LayerState

The :class:`~glue.viewers.common.state.LayerState` class defines the following
properties by default:

* ``layer``: the :class:`~glue.core.data.Data` or :class:`~glue.core.subset.Subset`
  attached to the layer (the naming of this property is historical/confusing and
  may be changed to ``data`` in future).
* ``visible``: whether the layer is visible or not
* ``zorder``: a numerical value indicating (when relevant) which layer should
  appear in front of which (higher numbers mean the layer should be shown more
  in the foreground)

Furthermore, ``layer.style`` is itself a state class that includes global
settings for the data or subset, such as ``color`` and ``alpha``.

Let's say that you want to define a way to indicate in the layer whether to
use filled markers or not - this is not one of the settings in ``layer.style``,
so you can define it using::

    class TutorialLayerState(LayerState):
        fill = CallbackProperty(False, docstring='Whether to show the markers as filled or not')

The optional first value in :class:`~echo.CallbackProperty` is the
default value that the property should be set to.

Multi-choice properties
^^^^^^^^^^^^^^^^^^^^^^^

In some cases, you might want the properties on the state classes to be a
selection from a fixed set of values -- for instance line style, or as
demonstrated in `Viewer State`_, the attribute to show on an axis (since
it should be chosen from the existing data attributes). This can be
done by using the :class:`~echo.SelectionCallbackProperty` class,
which should be used as follows::

    class TutorialViewerState(ViewerState):

        linestyle = SelectionCallbackProperty()

        def __init__(self, *args, **kwargs):
            super(TutorialViewerState).__init__(*args, **kwargs)
            MyExampleState.linestyle.set_choices(['solid', 'dashed', 'dotted'])

This then makes it so that the ``linestyle`` property knows about what valid
values are, and this will come in useful when developing for example Qt widgets
so that they can automatically  populate combo/selection boxes for example.

For the specific case of selecting attributes from the data, we also provide a
class :class:`~glue.core.data_combo_helper.ComponentIDComboHelper` that can
automatically keep the attributes for datasets in sync with the choices in a
:class:`~echo.SelectionCallbackProperty` class. Here's an example
of how to use it::

    class TutorialViewerState(ViewerState):

        x_att = SelectionCallbackProperty(docstring='The attribute to use on the x-axis')
        y_att = SelectionCallbackProperty(docstring='The attribute to use on the y-axis')

        def __init__(self, *args, **kwargs):
            super(TutorialViewerState, self).__init__(*args, **kwargs)
            self._x_att_helper = ComponentIDComboHelper(self, 'x_att')
            self._y_att_helper = ComponentIDComboHelper(self, 'y_att')
            self.add_callback('layers', self._on_layers_change)

        def _on_layers_change(self, value):
            # self.layers_data is a shortcut for
            # [layer_state.layer for layer_state in self.layers]
            self._x_att_helper.set_multiple_data(self.layers_data)
            self._y_att_helper.set_multiple_data(self.layers_data)

Now whenever layers are added/removed, the choices for ``x_att`` and ``y_att``
will automatically be updated.

Layer artist
------------

In the previous section, we saw that we can define classes to hold the
conceptual state of viewers and of the layers in the viewers. The next
type of class we are going to look at is the *layer artist*.

Conceptually, layer artists can be used to carry out the actual drawing and
include any logic about how to convert data and subsets into layers in your
visualization.

The minimal layer artist class looks like the following::

    from glue.viewers.common.layer_artist import LayerArtist

    class TutorialLayerArtist(LayerArtist):

        _layer_artist_cls = TutorialLayerState

        def clear(self):
            pass

        def remove(self):
            pass

        def redraw(self):
            pass

        def update(self):
            pass

Each layer artist class has to define the four methods shown above. The
:meth:`~glue.viewers.common.layer_artist.LayerArtist.clear` method
should remove the layer from the visualization, bearing in mind
that the layer might be added back (this can happen for example when toggling
the visibility of the layer property), the
:meth:`~glue.viewers.common.layer_artist.LayerArtist.remove` method
should permanently remove the layer from the visualization, the
:meth:`~glue.viewers.common.layer_artist.LayerArtist.redraw` method
should force the layer to be redrawn, and
:meth:`~glue.viewers.common.layer_artist.LayerArtist.update` should
update the appearance of the layer as necessary before redrawing -- note that
:meth:`~glue.viewers.common.layer_artist.LayerArtist.update` is called
for example when a subset has changed.

By default, layer artists inheriting from
:class:`~glue.viewers.common.layer_artist.LayerArtist` will be
initialized with a reference to the layer state (accessible as ``state``) and
the viewer state (accessible as ``_viewer_state``).

This means that we can then do the following, assuming a layer state
with the ``fill`` property defined previously::

  from glue.viewers.common.layer_artist import LayerArtist

  class TutorialLayerArtist(LayerArtist):

      _layer_artist_cls = TutorialLayerState

      def __init__(self, *args, **kwargs):
          super(MyLayerArtist, self).__init__(*args, **kwargs)
          self.state.add_callback('fill', self._on_fill_change)

      def _on_fill_change(self):
          # Make adjustments to the visualization layer here

In practice, you will likely need a reference to the overall visualization to
be passed to the layer artist (for example the axes for a Matplotlib plot,
or an OpenGL canvas). We will take a look at this after introducing the data
viewer class in `Data viewer`_.

Note that the layer artist doesn't have to be specific to the front-end used
either. If for instance you are developing a widget based on e.g.
Matplotlib, and are then developing a Qt and Jupyter version of the viewer,
you could write the layer artist in such a way that it only cares about the
Matplotlib API and works for either the Qt or Jupyter viewers.

Data viewer
-----------

We have now seen how to define state classes for the viewer and layer, and layer
artists. The final piece of the puzzle is the data viewer class itself, which
brings everything together. The simplest definition of the data viewer class
is::

    from glue.viewers.common.viewer import Viewer

    class TutorialDataViewer(Viewer):

        LABEL = 'Tutorial viewer'
        _state_cls = TutorialViewerState
        _data_artist_cls = TutorialLayerArtist
        _subset_artist_cls = TutorialLayerArtist

In practice, this isn't enough, since we need to actually set up the main
visualization and pass references to it to the layer artists. This can be
done in the initializer of the ``TutorialDataViewer`` class. For example,
if you were building a Matplotlib-based viewer, assuming you imported Matplotlib
as::

    from matplotlib import pyplot as plt

you could do::

    def __init__(self, *args, **kwargs):
        super(TutorialDataViewer, self).__init__(*args, **kwargs)
        self.axes = plt.subplot(1, 1, 1)

Note however that you need a way to pass the axes to the layer artist. The way
to do this is to add ``axes`` as a positional argument for the
``TutorialLayerArtist`` class defined previously then to add the following
method to the data viewer::

    def get_layer_artist(self, cls, layer=None, layer_state=None):
        return cls(self.axes, self.state, layer=layer, layer_state=layer_state)

This method defines how the layer artists should be instantiated, and you can
see that we added a ``self.axes`` positional argument, so that the layer artist
classes should now have access to the axes.

With this in place, what will happen now is that when a data viewer is created,
and when a new dataset or subset is added to it, the ``layers`` attribute of
the viewer state class will automatically be updated to include a new
:class:`~glue.viewers.common.state.LayerState` object. At the same time,
a :class:`~glue.viewers.common.layer_artist.LayerArtist` object will be
instantiated. The main task is therefore to implement the methods for the
:class:`~glue.viewers.common.layer_artist.LayerArtist` (in particular
:meth:`~glue.viewers.common.layer_artist.LayerArtist.update`). You can then add
any required logic in the state classes if needed.

Further reading
---------------

If you are interested in building a viewer for the Qt front-end of glue, you can
find out more about this and see a complete example in :ref:`state-qt-viewer`.
Even if you want to develop a viewer for a different front-end, you may find
the Qt example useful.
