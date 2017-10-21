"""
LayerArtist classes handle the visualization of an individual subset
or dataset.

Visualization clients in Glue typically combose visualizations by stacking
visualizations of several datasets and subsets on top of each other. They
do this by creating and managing a collection of LayerArtists, one for
each Data or Subset to view.

LayerArtists contain the bulk of the logic for actually rendering things
"""

from __future__ import absolute_import, division, print_function

import os
from contextlib import contextmanager
from abc import ABCMeta, abstractmethod

import numpy as np

from glue.external import six
from glue.core.subset import Subset
from glue.utils import Pointer, PropertySetMixin
from glue.core.message import LayerArtistEnabledMessage, LayerArtistDisabledMessage

__all__ = ['LayerArtistBase', 'MatplotlibLayerArtist', 'LayerArtistContainer']


DISABLED_LAYER_WARNING = """
This layer depends on attributes that cannot be derived for the underlying
dataset. This usually indicates that this dataset has not been linked with other
datasets being shown. In this case, for this layer to work, it would need to be
linked with the following datasets: {}
""".replace(os.linesep, ' ')

DISABLED_MASK_MESSAGE = """
The subset mask for this layer cannot be computed. This usually indicates that
the selection was defined using attributes that are not defined in this dataset.
""".replace(os.linesep, ' ')


class ChangedTrigger(object):

    """Sets an instance's _changed attribute to True on update"""

    def __init__(self, default=None):
        self._default = default
        self._vals = {}

    def __get__(self, inst, type=None):
        return self._vals.get(inst, self._default)

    def __set__(self, inst, value):
        if isinstance(value, np.ndarray):
            changed = value is not self.__get__(inst)
        else:
            changed = value != self.__get__(inst)
        self._vals[inst] = value
        if changed:
            inst._changed = True


@six.add_metaclass(ABCMeta)
class LayerArtistBase(PropertySetMixin):
    _property_set = ['zorder', 'visible', 'layer']

    # the order of this layer in the visualizations. High-zorder
    # layers are drawn on top of low-zorder layers.
    # Subclasses should refresh plots when this property changes
    zorder = Pointer('_zorder')

    # whether this layer should be rendered.
    # Subclasses should refresh plots when this property changes
    visible = Pointer('_visible')

    # whether this layer is capable of being rendered
    # Subclasses should refresh plots when this property changes
    enabled = Pointer('_enabled')

    def __init__(self, layer):
        """Create a new LayerArtist

        Parameters
        ----------
        layer : :class:`~glue.core.data.Data` or :class:`~glue.core.subset.Subset`
            Data or Subset to draw
        layer : :class:`~glue.core.data.Data` or `glue.core.subset.Subset`
        """
        self._visible = True
        self._zorder = 0
        self._enabled = True
        self._layer = layer

        self.view = None      # cache of last view, if relevant
        self._state = None    # cache of subset state, if relevant
        self._changed = True  # hint at whether underlying data has changed since last render

        self._disabled_reason = ''  # A string explaining why this layer is disabled.

    def get_layer_color(self):
        # This method can return either a plain color or a colormap. This is
        # used by the UI layer to determine a 'representative' color or colormap
        # for the layer to be used e.g. in icons.
        return self._layer.style.color

    def enable(self):
        if self.enabled:
            return
        self._disabled_reason = ''
        self._enabled = True
        self.redraw()
        if self._layer is not None and self._layer.hub is not None:
            message = LayerArtistEnabledMessage(self)
            self._layer.hub.broadcast(message)

    def disable(self, reason):
        """
        Disable the layer for a particular reason.

        Layers should only be disabled when drawing is impossible,
        e.g. because a subset cannot be applied to a dataset.

        Parameters
        ----------
        reason : str
           A short explanation for why the layer can't be drawn.
           Used by the UI
        """
        self._disabled_reason = reason
        self._enabled = False
        self.clear()
        if self._layer is not None and self._layer.hub is not None:
            message = LayerArtistDisabledMessage(self)
            self._layer.hub.broadcast(message)

    def disable_invalid_attributes(self, *attributes):
        """
        Disable a layer because visualization depends on knowing a set
        of ComponentIDs that cannot be derived from a dataset or subset

        Automatically generates a disabled message.

        Parameters
        ----------
        attributes : sequence of ComponentIDs
        """
        if len(attributes) == 0:
            self.disable('')
            return

        datasets = ', '.join(sorted(set([cid.parent.label for cid in attributes])))
        self.disable(DISABLED_LAYER_WARNING.format(datasets))

    def disable_incompatible_subset(self):
        """
        Disable a layer because the subset mask cannot be computed.

        Automatically generates a disabled message.
        """
        self.disable(DISABLED_MASK_MESSAGE)

    @property
    def disabled_message(self):
        """
        Returns why a layer is disabled
        """
        if self.enabled:
            return ''
        return "Cannot visualize this layer: %s" % self._disabled_reason

    @property
    def layer(self):
        """
        The Data or Subset visualized in this layer
        """
        return self._layer

    @layer.setter
    def layer(self, value):
        self._layer = value

    @abstractmethod
    def redraw(self):
        """
        Re-render the plot
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self, view=None):
        """
        Sync the visual appearance of the layer, and redraw

        Subclasses may skip the update if the _changed attribute
        is set to False.

        Parameters
        ----------
        view : (ComponentID, numpy_style view) or None
            A hint about what sub-view into the data is relevant.
        """
        raise NotImplementedError()

    @abstractmethod
    def clear(self):
        """Clear the visualization for this layer"""
        raise NotImplementedError()

    def remove(self):
        """
        Remove the visualization for this layer.

        This is called when the layer artist is removed for good from the
        viewer. It defaults to calling clear, but can be overriden in cases
        where clear and remove should be different.
        """
        self.clear()

    def force_update(self, *args, **kwargs):
        """
        Sets the _changed flag to true, and calls update.

        Force an update of the layer, overriding any
        caching that might be going on for speed
        """
        self._changed = True
        return self.update(*args, **kwargs)

    def _check_subset_state_changed(self):
        """Checks to see if layer is a subset and, if so,
        if it has changed subset state. Sets _changed flag to True if so"""
        if not isinstance(self.layer, Subset):
            return
        state = self.layer.subset_state
        if state is not self._state:
            self._changed = True
            self._state = state

    def __str__(self):
        return "%s for %s" % (self.__class__.__name__, self.layer.label)

    def __gluestate__(self, context):
        # note, this doesn't yet have a restore method. Will rely on client
        return dict((k, context.id(v)) for k, v in self.properties.items())

    __repr__ = __str__


class MatplotlibLayerArtist(LayerArtistBase):

    """
    MPL-specific layer artist base class, that uses an Axes object
    """

    def __init__(self, layer, axes):
        super(MatplotlibLayerArtist, self).__init__(layer)
        self._axes = axes
        self.artists = []

    def redraw(self):
        self._axes.figure.canvas.draw()

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        self._visible = value
        for a in self.artists:
            a.set_visible(value)

    def _sync_style(self):
        style = self.layer.style
        for artist in self.artists:
            edgecolor = style.color
            # due to a bug in MPL 1.4.1, we can't disable the edge
            # without making the whole point disappear. So we make the
            # edge very thin instead
            mew = 3 if style.marker == '+' else 0.01
            artist.set_markeredgecolor(edgecolor)
            artist.set_markeredgewidth(mew)
            artist.set_markerfacecolor(style.color)
            artist.set_marker(style.marker)
            artist.set_markersize(style.markersize)
            artist.set_linestyle('None')
            artist.set_alpha(style.alpha)
            artist.set_zorder(self.zorder)
            artist.set_visible(self.visible and self.enabled)

    @property
    def zorder(self):
        return self._zorder

    @zorder.setter
    def zorder(self, value):
        for artist in self.artists:
            artist.set_zorder(value)
        self._zorder = value

    @property
    def enabled(self):
        return len(self.artists) > 0

    def clear(self):
        for artist in self.artists:
            try:
                artist.remove()
            except ValueError:  # already removed
                pass
        self.artists = []


class LayerArtistContainer(object):

    """A collection of LayerArtists"""

    def __init__(self):
        self.artists = []
        self.empty_callbacks = []
        self.change_callbacks = []
        self._ignore_callbacks = False

    def on_empty(self, func):
        """
        Register a callback function that should be invoked when
        this container is emptied
        """
        self.empty_callbacks.append(func)

    def on_changed(self, func):
        """
        Register a callback function that should be invoked when
        this container's elements change
        """
        self.change_callbacks.append(func)

    def _duplicate(self, artist):
        for a in self.artists:
            if type(a) == type(artist) and a.layer is artist.layer:
                return True
        return False

    def append(self, artist):
        """Add a LayerArtist to this collection"""
        self.artists.append(artist)
        artist.zorder = max(a.zorder for a in self.artists) + 1
        self._notify()

    def remove(self, artist):
        """Remove a LayerArtist from this collection

        :param artist: The artist to remove
        :type artist: :class:`MatplotlibLayerArtist`
        """
        if artist in self.artists:
            self.artists.remove(artist)
            artist.remove()
            self._notify()

    def clear(self):
        """
        Remove all layer artists from this collection
        """
        for artist in self.artists:
            artist.remove()
        if six.PY2:
            self.artists[:] = []
        else:
            self.artists.clear()

    def clear_callbacks(self):
        """
        Remove all callbacks
        """
        if six.PY2:
            self.empty_callbacks[:] = []
            self.change_callbacks[:] = []
        else:
            self.empty_callbacks.clear()
            self.change_callbacks.clear()

    def _notify(self):
        if self._ignore_callbacks:
            return

        for cb in self.change_callbacks:
            cb()

        if len(self) == 0:
            for cb in self.empty_callbacks:
                cb()

    def pop(self, layer):
        """Remove all artists associated with a layer"""
        to_remove = [a for a in self.artists if a.layer is layer]
        for r in to_remove:
            self.remove(r)
        return to_remove

    @property
    def layers(self):
        """A list of the unique layers in the container"""
        return list(set([a.layer for a in self.artists]))

    @contextmanager
    def ignore_empty(self):
        """A context manager that temporarily disables calling callbacks if container is emptied"""
        try:
            self._ignore_callbacks = True
            yield
        finally:
            self._ignore_callbacks = False

    def __len__(self):
        return len(self.artists)

    def __iter__(self):
        return iter(sorted(self.artists, key=lambda x: x.zorder))

    def __contains__(self, item):
        if isinstance(item, MatplotlibLayerArtist):
            return item in self.artists
        return any(item is a.layer for a in self.artists)

    def __getitem__(self, layer):
        if isinstance(layer, int):
            return self.artists[layer]
        return [a for a in self.artists if a.layer is layer]
