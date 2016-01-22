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

import logging
from contextlib import contextmanager
from abc import ABCMeta, abstractproperty, abstractmethod

import numpy as np
from matplotlib.cm import gray

from glue.external import six
from glue.core.subset import Subset
from glue.core.exceptions import IncompatibleAttribute
from glue.clients.ds9norm import DS9Normalize
from glue.clients.util import small_view, small_view_array
from glue.utils import (view_cascade, get_extent, color2rgb, Pointer,
                        PropertySetMixin)


__all__ = ['LayerArtistBase', 'LayerArtist',
           'HistogramLayerArtist', 'ScatterLayerArtist',
           'LayerArtistContainer', 'RGBImageLayerArtist', 'ImageLayerArtist']


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

        msg = ('Layer depends on attributes that '
               'cannot be derived for %s:\n -%s' %
               (self._layer.data.label,
                '\n -'.join(map(str, attributes))))

        self.disable(msg)

    @property
    def disabled_message(self):
        """
        Returns why a layer is disabled
        """
        if self.enabled:
            return ''
        return "Cannot visualize this layer\n%s" % self._disabled_reason

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
        """Clear the visulaization for this layer"""
        raise NotImplementedError()

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


"""
Base-class mixin interfaces for different visualizations.
"""


@six.add_metaclass(ABCMeta)
class ScatterLayerBase(object):

    # which ComponentID to assign to X axis
    xatt = abstractproperty()

    # which ComponentID to assign to Y axis
    yatt = abstractproperty()

    @abstractmethod
    def get_data(self):
        """
        Returns
        -------
        array
            The scatterpoint data as an (N, 2) array
        """
        pass


@six.add_metaclass(ABCMeta)
class RGBImageLayerBase(object):

    r = abstractproperty()               # ComponentID for red channel
    g = abstractproperty()               # ComponentID for green channel
    b = abstractproperty()               # ComponentID for blue channel
    rnorm = abstractproperty()           # Normalize instance for red channel
    gnorm = abstractproperty()           # Normalize instance for green channel
    bnorm = abstractproperty()  # Normalize instance for blue channel
    contrast_layer = abstractproperty()  # 'red' | 'green' | 'blue'. Which norm to adjust during set_norm
    layer_visible = abstractproperty()   # dict (str->bool). Whether to show 'red', 'green', 'blue' layers

    @property
    def color_visible(self):
        """
        Return layer visibility as a list of [red_visible, green_visible, blue_visible]
        """
        return [self.layer_visible['red'], self.layer_visible['green'],
                self.layer_visible['blue']]

    @color_visible.setter
    def color_visible(self, value):
        self.layer_visible['red'] = value[0]
        self.layer_visible['green'] = value[1]
        self.layer_visible['blue'] = value[2]


@six.add_metaclass(ABCMeta)
class HistogramLayerBase(object):
    lo = abstractproperty()     # lo-cutoff for bin counting
    hi = abstractproperty()     # hi-cutoff for bin counting
    nbins = abstractproperty()  # number of bins
    xlog = abstractproperty()   # whether to space bins logarithmically

    @abstractmethod
    def get_data(self):
        """
        Return array of bin counts
        """
        pass


@six.add_metaclass(ABCMeta)
class ImageLayerBase(object):

    norm = abstractproperty()  # Normalization instance to scale intensities
    cmap = abstractproperty()  # colormap

    @abstractmethod
    def set_norm(self, **kwargs):
        """
        Adjust the normalization instance parameters.
        See :class:`glue.clients.ds9norm.DS9Normalize attributes for valid
        kwargs for this function
        """
        pass

    @abstractmethod
    def clear_norm():
        """
        Reset the norm to the default
        """
        pass

    @abstractmethod
    def override_image(self, image):
        """
        Temporarily display another image instead of a view into the data

        The new image has the same shape as the view into the data
        """
        pass

    @abstractmethod
    def clear_override(self):
        """
        Remove the override image, and display the data again
        """
        pass


@six.add_metaclass(ABCMeta)
class SubsetImageLayerBase(object):
    pass


"""
Matplotlib-specific implementations follow
"""


class LayerArtist(LayerArtistBase):

    """
    MPL-specific layer artist base class, that uses an Axes object
    """

    def __init__(self, layer, axes):
        super(LayerArtist, self).__init__(layer)
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


class ImageLayerArtist(LayerArtist, ImageLayerBase):
    _property_set = LayerArtist._property_set + ['norm']

    def __init__(self, layer, ax):
        super(ImageLayerArtist, self).__init__(layer, ax)
        self._norm = None
        self._cmap = gray
        self._override_image = None
        self._clip_cache = None
        self.aspect = 'equal'

    @property
    def norm(self):
        return self._norm

    @norm.setter
    def norm(self, value):
        self._norm = value

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, value):
        self._cmap = value
        for a in self.artists:
            a.set_cmap(value)

    def _default_norm(self, layer):
        vals = np.sort(layer.ravel())
        vals = vals[np.isfinite(vals)]
        result = DS9Normalize()
        result.stretch = 'arcsinh'
        result.clip = True
        if vals.size > 0:
            result.vmin = vals[np.intp(.01 * vals.size)]
            result.vmax = vals[np.intp(.99 * vals.size)]
        return result

    def override_image(self, image):
        """Temporarily show a different image"""
        self._override_image = image

    def clear_override(self):
        self._override_image = None

    def _extract_view(self, view, transpose):
        if self._override_image is None:
            result = self.layer[view]
            if transpose:
                result = result.T
            return result
        else:
            v = [v for v in view if isinstance(v, slice)]
            if transpose:
                v = v[::-1]
            result = self._override_image[v]
            return result

    def _update_clip(self, att):
        key = (att, id(self._override_image),
               self.norm.clip_lo, self.norm.clip_hi)
        if self._clip_cache == key:
            return
        self._clip_cache = key

        if self._override_image is None:
            data = small_view(self.layer, att)
        else:
            data = small_view_array(self._override_image)
        self.norm.update_clip(data)

    def update(self, view, transpose=False, aspect=None):

        if aspect is not None:
            self.aspect = aspect

        self.clear()
        views = view_cascade(self.layer, view)
        artists = []

        lr0 = self._extract_view(views[0], transpose)
        self.norm = self.norm or self._default_norm(lr0)
        self.norm = self.norm or self._default_norm(lr0)
        self._update_clip(views[0][0])

        for v in views:
            image = self._extract_view(v, transpose)
            extent = get_extent(v, transpose)
            artists.append(self._axes.imshow(image, cmap=self.cmap,
                                             norm=self.norm,
                                             interpolation='nearest',
                                             origin='lower',
                                             extent=extent, zorder=0))
            self._axes.set_aspect(self.aspect, adjustable='datalim')
        self.artists = artists
        self._sync_style()

    def set_norm(self, vmin=None, vmax=None,
                 bias=None, contrast=None, stretch=None, norm=None,
                 clip_lo=None, clip_hi=None):
        if norm is not None:
            self.norm = norm  # XXX Should wrap ala DS9Normalize(norm)
            return norm
        if self.norm is None:
            self.norm = DS9Normalize()
        if vmin is not None:
            self.norm.vmin = vmin
        if vmax is not None:
            self.norm.vmax = vmax
        if bias is not None:
            self.norm.bias = bias
        if contrast is not None:
            self.norm.contrast = contrast
        if clip_lo is not None:
            self.norm.clip_lo = clip_lo
        if clip_hi is not None:
            self.norm.clip_hi = clip_hi
        if stretch is not None:
            self.norm.stretch = stretch
        return self.norm

    def clear_norm(self):
        self.norm = None

    def _sync_style(self):
        for artist in self.artists:
            artist.set_zorder(self.zorder)
            artist.set_visible(self.visible and self.enabled)


class RGBImageLayerArtist(ImageLayerArtist, RGBImageLayerBase):
    _property_set = ImageLayerArtist._property_set + \
        ['r', 'g', 'b', 'rnorm', 'gnorm', 'bnorm', 'color_visible']

    r = ChangedTrigger()
    g = ChangedTrigger()
    b = ChangedTrigger()
    rnorm = Pointer('_rnorm')
    gnorm = Pointer('_gnorm')
    bnorm = Pointer('_bnorm')

    # dummy class-level variables will be masked
    # at instance level, needed for ABC to be happy
    layer_visible = None
    contrast_layer = None

    def __init__(self, layer, ax, last_view=None):
        super(RGBImageLayerArtist, self).__init__(layer, ax)
        self.contrast_layer = 'green'
        self.aspect = 'equal'
        self.layer_visible = dict(red=True, green=True, blue=True)
        self.last_view = last_view

    def set_norm(self, *args, **kwargs):
        spr = super(RGBImageLayerArtist, self).set_norm
        if self.contrast_layer == 'red':
            self.norm = self.rnorm
            self.rnorm = spr(*args, **kwargs)
        if self.contrast_layer == 'green':
            self.norm = self.gnorm
            self.gnorm = spr(*args, **kwargs)
        if self.contrast_layer == 'blue':
            self.norm = self.bnorm
            self.bnorm = spr(*args, **kwargs)

    def update(self, view=None, transpose=False, aspect=None):

        self.clear()

        if aspect is not None:
            self.aspect = aspect

        if self.r is None or self.g is None or self.b is None:
            return

        if view is None:
            view = self.last_view

        if view is None:
            return
        self.last_view = view

        views = view_cascade(self.layer, view)
        artists = []
        for v in views:
            extent = get_extent(v, transpose)
            # first argument = component. swap
            r = tuple([self.r] + list(v[1:]))
            g = tuple([self.g] + list(v[1:]))
            b = tuple([self.b] + list(v[1:]))
            r = self.layer[r]
            g = self.layer[g]
            b = self.layer[b]
            if transpose:
                r = r.T
                g = g.T
                b = b.T
            self.rnorm = self.rnorm or self._default_norm(r)
            self.gnorm = self.gnorm or self._default_norm(g)
            self.bnorm = self.bnorm or self._default_norm(b)
            if v is views[0]:
                self.rnorm.update_clip(small_view(self.layer, self.r))
                self.gnorm.update_clip(small_view(self.layer, self.g))
                self.bnorm.update_clip(small_view(self.layer, self.b))

            image = np.dstack((self.rnorm(r),
                               self.gnorm(g),
                               self.bnorm(b)))

            if not self.layer_visible['red']:
                image[:, :, 0] *= 0
            if not self.layer_visible['green']:
                image[:, :, 1] *= 0
            if not self.layer_visible['blue']:
                image[:, :, 2] *= 0

            artists.append(self._axes.imshow(image,
                                             interpolation='nearest',
                                             origin='lower',
                                             extent=extent, zorder=0))
            self._axes.set_aspect(self.aspect, adjustable='datalim')
        self.artists = artists
        self._sync_style()


class SubsetImageLayerArtist(LayerArtist, SubsetImageLayerBase):

    def __init__(self, *args, **kwargs):
        super(SubsetImageLayerArtist, self).__init__(*args, **kwargs)
        self.aspect = 'equal'

    def update(self, view, transpose=False, aspect=None):

        self.clear()

        if aspect is not None:
            self.aspect = aspect

        subset = self.layer
        logging.debug("View into subset %s is %s", self.layer, view)

        try:
            mask = subset.to_mask(view[1:])
        except IncompatibleAttribute as exc:
            self.disable_invalid_attributes(*exc.args)
            return False
        logging.debug("View mask has shape %s", mask.shape)

        # shortcut for empty subsets
        if not mask.any():
            return

        if transpose:
            mask = mask.T

        extent = get_extent(view, transpose)
        r, g, b = color2rgb(self.layer.style.color)
        mask = np.dstack((r * mask, g * mask, b * mask, mask * .5))
        mask = (255 * mask).astype(np.uint8)
        self.artists = [self._axes.imshow(mask, extent=extent,
                                          interpolation='nearest',
                                          origin='lower',
                                          zorder=5, visible=self.visible)]
        self._axes.set_aspect(self.aspect, adjustable='datalim')


class ScatterLayerArtist(LayerArtist, ScatterLayerBase):
    xatt = ChangedTrigger()
    yatt = ChangedTrigger()
    _property_set = LayerArtist._property_set + ['xatt', 'yatt']

    def __init__(self, layer, ax):
        super(ScatterLayerArtist, self).__init__(layer, ax)
        self.emphasis = None  # an optional SubsetState of emphasized points

    def _recalc(self):
        self.clear()
        assert len(self.artists) == 0

        try:
            x = self.layer[self.xatt].ravel()
            y = self.layer[self.yatt].ravel()
        except IncompatibleAttribute as exc:
            self.disable_invalid_attributes(*exc.args)
            return False

        self.artists = self._axes.plot(x, y)
        return True

    def update(self, view=None, transpose=False):
        self._check_subset_state_changed()

        if self._changed:  # erase and make a new artist
            if not self._recalc():  # no need to update style
                return
            self._changed = False

        has_emph = False
        if self.emphasis is not None:
            try:
                s = Subset(self.layer.data)
                s.subset_state = self.emphasis
                if hasattr(self.layer, 'subset_state'):
                    s.subset_state &= self.layer.subset_state
                x = s[self.xatt].ravel()
                y = s[self.yatt].ravel()
                self.artists.extend(self._axes.plot(x, y))
                has_emph = True
            except IncompatibleAttribute:
                pass

        self._sync_style()
        if has_emph:
            self.artists[-1].set_mec('green')
            self.artists[-1].set_mew(2)
            self.artists[-1].set_alpha(1)

    def get_data(self):
        try:
            return self.layer[self.xatt].ravel(), self.layer[self.yatt].ravel()
        except IncompatibleAttribute:
            return np.array([]), np.array([])


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

    def _check_duplicate(self, artist):
        """Raise an error if this artist is a duplicate"""
        if self._duplicate(artist):
            raise ValueError("Already have an artist for this type "
                             "and data")

    def append(self, artist):
        """Add a LayerArtist to this collection"""
        self._check_duplicate(artist)
        self.artists.append(artist)
        artist.zorder = max(a.zorder for a in self.artists) + 1
        self._notify()

    def remove(self, artist):
        """Remove a LayerArtist from this collection

        :param artist: The artist to remove
        :type artist: :class:`LayerArtist`
        """
        try:
            self.artists.remove(artist)
            artist.clear()
        except ValueError:
            pass

        self._notify()

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
        if isinstance(item, LayerArtist):
            return item in self.artists
        return any(item is a.layer for a in self.artists)

    def __getitem__(self, layer):
        if isinstance(layer, int):
            return self.artists[layer]
        return [a for a in self.artists if a.layer is layer]


class HistogramLayerArtist(LayerArtist, HistogramLayerBase):
    _property_set = LayerArtist._property_set + 'lo hi nbins xlog'.split()

    lo = ChangedTrigger(0)
    hi = ChangedTrigger(1)
    nbins = ChangedTrigger(10)
    xlog = ChangedTrigger(False)

    def __init__(self, layer, axes):
        super(HistogramLayerArtist, self).__init__(layer, axes)
        self.ylog = False
        self.cumulative = False
        self.normed = False
        self.y = np.array([])
        self.x = np.array([])
        self._y = np.array([])

        self._scale_state = None

    def get_data(self):
        return self.x, self.y

    def clear(self):
        super(HistogramLayerArtist, self).clear()
        self.x = np.array([])
        self.y = np.array([])
        self._y = np.array([])

    def _calculate_histogram(self):
        """Recalculate the histogram, creating new patches"""
        self.clear()
        try:
            data = self.layer[self.att].ravel()
            if not np.isfinite(data).any():
                return False
        except IncompatibleAttribute as exc:
            self.disable_invalid_attributes(*exc.args)
            return False

        if data.size == 0:
            return

        if self.lo > np.nanmax(data) or self.hi < np.nanmin(data):
            return
        if self.xlog:
            data = np.log10(data)
            rng = [np.log10(self.lo), np.log10(self.hi)]
        else:
            rng = self.lo, self.hi
        nbinpatch = self._axes.hist(data,
                                    bins=int(self.nbins),
                                    range=rng)
        self._y, self.x, self.artists = nbinpatch
        return True

    def _scale_histogram(self):
        """Modify height of bins to match ylog, cumulative, and norm"""
        if self.x.size == 0:
            return

        y = self._y.astype(np.float)
        dx = self.x[1] - self.x[0]
        if self.normed:
            div = y.sum() * dx
            if div == 0:
                div = 1
            y /= div
        if self.cumulative:
            y = y.cumsum()
            y /= y.max()

        self.y = y
        bottom = 0 if not self.ylog else 1e-100

        for a, y in zip(self.artists, y):
            a.set_height(y)
            x, y = a.get_xy()
            a.set_xy((x, bottom))

    def _check_scale_histogram(self):
        """
        If needed, rescale histogram to match cumulative/log/normed state.
        """
        state = (self.normed, self.ylog, self.cumulative)
        if state == self._scale_state:
            return
        self._scale_state = state
        self._scale_histogram()

    def update(self, view=None):
        """Sync plot.

        The _change flag tracks whether the histogram needs to be
        recalculated. If not, the properties of the existing
        artists are updated
        """
        self._check_subset_state_changed()
        if self._changed:
            if not self._calculate_histogram():
                return
            self._changed = False
            self._scale_state = None
        self._check_scale_histogram()
        self._sync_style()

    def _sync_style(self):
        """Update visual properties"""
        style = self.layer.style
        for artist in self.artists:
            artist.set_facecolor(style.color)
            artist.set_alpha(style.alpha)
            artist.set_zorder(self.zorder)
            artist.set_visible(self.visible and self.enabled)
