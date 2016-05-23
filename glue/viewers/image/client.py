from __future__ import absolute_import, division, print_function

import logging
from functools import wraps

import numpy as np

from glue.external.modest_image import extract_matched_slices
from glue.core.edit_subset_mode import EditSubsetMode
from glue.core.callback_property import (
    callback_property, CallbackProperty)
from glue.core.message import ComponentReplacedMessage, SettingsChangeMessage
from glue.core.roi import PolygonalROI
from glue.core.subset import Subset, RoiSubsetState
from glue.core.data import Data
from glue.core.exceptions import IncompatibleAttribute
from glue.core.layer_artist import LayerArtistContainer
from glue.core.state import lookup_class_with_patches
from glue.utils import defer_draw

from glue.viewers.common.viz_client import VizClient, init_mpl, update_appearance_from_settings
from glue.viewers.scatter.layer_artist import ScatterLayerBase, ScatterLayerArtist

from .layer_artist import (ImageLayerArtist, SubsetImageLayerArtist,
                           RGBImageLayerArtist, ImageLayerBase,
                           RGBImageLayerBase, SubsetImageLayerBase)


def requires_data(func):
    """
    Decorator that checks an ImageClient for a non-null display_data
    attribute. Only executes decorated function if present.
    """
    @wraps(func)
    def result(*args, **kwargs):
        if args[0].display_data is None:
            return
        return func(*args, **kwargs)
    return result


class ImageClient(VizClient):

    display_data = CallbackProperty(None)
    display_attribute = CallbackProperty(None)
    display_aspect = CallbackProperty('equal')

    def __init__(self, data, layer_artist_container=None):

        VizClient.__init__(self, data)

        self.artists = layer_artist_container
        if self.artists is None:
            self.artists = LayerArtistContainer()

        # slice through ND cube
        # ('y', 'x', 2)
        # means current data slice is [:, :, 2], and axis=0 is vertical on plot
        self._slice = None

        # how to extract a downsampled/cropped 2D image to plot
        # (ComponentID, slice, slice, ...)
        self._view = None

        # cropped/downsampled image
        # self._image == self.display_data[self._view]
        self._image = None

        # if this is set, render this instead of self._image
        self._override_image = None

        # maps attributes -> normalization settings
        self._norm_cache = {}

    def point_details(self, x, y):
        if self.display_data is None:
            return dict(labels=['x=%s' % x, 'y=%s' % y],
                        pix=(x, y), world=(x, y), value=np.nan)

        data = self.display_data
        pix = self._pixel_coords(x, y)
        labels = self.coordinate_labels(pix)
        world = data.coords.pixel2world(*pix[::-1])
        world = world[::-1]  # reverse for numpy convention

        view = []
        for p, s in zip(pix, data.shape):
            p = int(p)
            if not (0 <= p < s):
                value = None
                break
            view.append(slice(p, p + 1))
        else:
            if self._override_image is None:
                value = self.display_data[self.display_attribute, view]
            else:
                value = self._override_image[int(y), int(x)]

            value = value.ravel()[0]

        return dict(pix=pix, world=world, labels=labels, value=value)

    def coordinate_labels(self, pix):
        """
        Return human-readable labels for a position in pixel coords

        Parameters
        ----------
        pix : tuple of int
            Pixel coordinates of point in the data. Note that pix describes a
            position in the *data*, not necessarily the image display.

        Returns
        -------
        list
            A list of strings for each coordinate axis, of the form
            ``axis_label_name=world_coordinate_value``
        """
        data = self.display_data
        if data is None:
            return []

        world = data.coords.pixel2world(*pix[::-1])
        world = world[::-1]   # reverse for numpy convention
        labels = ['%s=%s' % (data.get_world_component_id(i).label, w)
                  for i, w in enumerate(world)]
        return labels

    @callback_property
    def slice(self):
        """
        Returns a tuple describing the current slice through the data

        The tuple has length equal to the dimensionality of the display
        data. Each entry is either:

        * 'x' if the dimension is mapped to the X image axis
        * 'y' if the dimension is mapped to the Y image axis
        * a number, indicating which fixed slice the dimension is restricted to
        """
        if self._slice is not None:
            return self._slice

        if self.display_data is None:
            return tuple()
        ndim = self.display_data.ndim
        if ndim == 1:
            self._slice = ('x',)
        elif ndim == 2:
            self._slice = ('y', 'x')
        else:
            self._slice = (0,) * (ndim - 2) + ('y', 'x')

        return self._slice

    @slice.setter
    @defer_draw
    def slice(self, value):
        if self.slice == tuple(value):
            return

        if value == tuple():
            return

        relim = value.index('x') != self._slice.index('x') or \
            value.index('y') != self._slice.index('y')

        self._slice = tuple(value)
        self._clear_override()
        self._update_axis_labels()
        self._update_data_plot(relim=relim)
        self._update_subset_plots()
        self._update_scatter_plots()
        self._redraw()

    @property
    def is_3D(self):
        """
        Returns True if the display data has 3 dimensions
        """
        if not self.display_data:
            return False
        return len(self.display_data.shape) == 3

    @property
    def slice_ind(self):
        """
        For 3D data, returns the pixel index of the current slice.
        Otherwise, returns `None`.
        """
        if self.is_3D:
            for s in self.slice:
                if s not in ['x', 'y']:
                    return s
        return None

    @property
    def image(self):
        return self._image

    @requires_data
    def override_image(self, image):
        """
        Temporarily override the current slice view with another image (i.e.,
        an aggregate).
        """
        self._override_image = image
        for a in self.artists[self.display_data]:
            if isinstance(a, ImageLayerBase):
                a.override_image(image)
        self._update_data_plot()
        self._redraw()

    def _clear_override(self):
        self._override_image = None
        for a in self.artists[self.display_data]:
            if isinstance(a, ImageLayerBase):
                a.clear_override()

    @slice_ind.setter
    @defer_draw
    def slice_ind(self, value):
        if self.is_3D:
            slc = [s if s in ['x', 'y'] else value for s in self.slice]
            self.slice = slc
            self._update_data_plot()
            self._update_subset_plots()
            self._update_scatter_plots()
            self._redraw()
        else:
            raise IndexError("Can only set slice_ind for 3D images")

    def can_image_data(self, data):
        return data.ndim > 1

    def _ensure_data_present(self, data):
        if data not in self.artists:
            self.add_layer(data)

    @defer_draw
    def set_data(self, data, attribute=None):
        if not self.can_image_data(data):
            return

        self._ensure_data_present(data)
        self._slice = None

        attribute = attribute or _default_component(data)

        self.display_data = data
        self.display_attribute = attribute
        self._update_axis_labels()
        self._update_data_plot(relim=True)
        self._update_subset_plots()
        self._update_scatter_plots()
        self._redraw()

    def set_attribute(self, attribute):
        if not self.display_data or \
                attribute not in self.display_data.component_ids():
            raise IncompatibleAttribute(
                "Attribute not in data's attributes: %s" % attribute)
        if self.display_attribute is not None:
            self._norm_cache[self.display_attribute] = self.get_norm()

        self.display_attribute = attribute

        if attribute in self._norm_cache:
            self.set_norm(norm=self._norm_cache[attribute])
        else:
            self.clear_norm()

        self._update_data_plot()
        self._redraw()

    def _redraw(self):
        """
        Re-render the screen.
        """
        pass

    @requires_data
    @defer_draw
    def set_norm(self, **kwargs):
        for a in self.artists[self.display_data]:
            a.set_norm(**kwargs)
        self._update_data_plot()
        self._redraw()

    @requires_data
    def clear_norm(self):
        for a in self.artists[self.display_data]:
            a.clear_norm()

    @requires_data
    def get_norm(self):
        a = self.artists[self.display_data][0]
        return a.norm

    @requires_data
    @defer_draw
    def set_cmap(self, cmap):
        for a in self.artists[self.display_data]:
            a.cmap = cmap
            a.redraw()

    def _build_view(self):
        att = self.display_attribute
        shp = self.display_data.shape
        x, y = np.s_[:], np.s_[:]
        slc = list(self.slice)
        slc[slc.index('x')] = x
        slc[slc.index('y')] = y
        return (att,) + tuple(slc)

    @requires_data
    def _numerical_data_changed(self, message):
        data = message.sender
        self._update_data_plot(force=True)
        self._update_scatter_layer(data)

        for s in data.subsets:
            self._update_subset_single(s, force=True)

        self._redraw()

    @requires_data
    def _update_data_plot(self, relim=False, force=False):
        """
        Re-sync the main image and its subsets.
        """

        if relim:
            self.relim()

        view = self._build_view()
        self._image = self.display_data[view]
        transpose = self.slice.index('x') < self.slice.index('y')

        self._view = view
        for a in list(self.artists):
            if (not isinstance(a, ScatterLayerBase) and
                    a.layer.data is not self.display_data):
                self.artists.remove(a)
            else:
                if isinstance(a, ImageLayerArtist):
                    a.update(view, transpose, aspect=self.display_aspect)
                else:
                    a.update(view, transpose)
        for a in self.artists[self.display_data]:
            meth = a.update if not force else a.force_update
            if isinstance(a, ImageLayerArtist):
                meth(view, transpose=transpose, aspect=self.display_aspect)
            else:
                meth(view, transpose=transpose)

    def _update_subset_single(self, s, redraw=False, force=False):
        """
        Update the location and visual properties of each point in a single
        subset.

        Parameters
        ----------
        s: `~glue.core.subset.Subset`
            The subset to refresh.
        """
        logging.getLogger(__name__).debug("update subset single: %s", s)

        if s not in self.artists:
            return

        self._update_scatter_layer(s)

        if s.data is not self.display_data:
            return

        view = self._build_view()
        transpose = self.slice.index('x') < self.slice.index('y')
        for a in self.artists[s]:
            meth = a.update if not force else a.force_update
            if isinstance(a, SubsetImageLayerArtist):
                meth(view, transpose=transpose, aspect=self.display_aspect)
            else:
                meth(view, transpose=transpose)

        if redraw:
            self._redraw()

    @property
    def _slice_ori(self):
        if not self.is_3D:
            return None
        for i, s in enumerate(self.slice):
            if s not in ['x', 'y']:
                return i

    @requires_data
    @defer_draw
    def apply_roi(self, roi):

        subset_state = RoiSubsetState()
        xroi, yroi = roi.to_polygon()
        x, y = self._get_plot_attributes()
        subset_state.xatt = x
        subset_state.yatt = y
        subset_state.roi = PolygonalROI(xroi, yroi)
        mode = EditSubsetMode()
        mode.update(self.data, subset_state, focus_data=self.display_data)

    def _remove_subset(self, message):
        self.delete_layer(message.sender)

    def delete_layer(self, layer):
        if layer not in self.artists:
            return
        for a in self.artists.pop(layer):
            a.clear()

        if isinstance(layer, Data):
            for subset in layer.subsets:
                self.delete_layer(subset)

        if layer is self.display_data:
            for layer in self.artists:
                if isinstance(layer, ImageLayerArtist):
                    self.display_data = layer.data
                    break
            else:
                for artist in self.artists:
                    self.delete_layer(artist.layer)
                self.display_data = None
                self.display_attribute = None

        self._redraw()

    def _remove_data(self, message):
        self.delete_layer(message.data)
        for s in message.data.subsets:
            self.delete_layer(s)

    def init_layer(self, layer):
        # only auto-add subsets if they are of the main image
        if isinstance(layer, Subset) and layer.data is not self.display_data:
            return
        self.add_layer(layer)

    def rgb_mode(self, enable=None):
        """
        Query whether RGB mode is enabled, or toggle RGB mode.

        Parameters
        ----------
        enable : bool or None
            If `True` or `False`, explicitly enable/disable RGB mode.
            If `None`, check if RGB mode is enabled

        Returns
        -------
        LayerArtist or None
            If RGB mode is enabled, returns an ``RGBImageLayerBase``.
            If ``enable`` is `False`, return the new ``ImageLayerArtist``
        """
        # XXX need to better handle case where two RGBImageLayerArtists
        #    are created

        if enable is None:
            for a in self.artists:
                if isinstance(a, RGBImageLayerBase):
                    return a
            return None

        result = None
        layer = self.display_data
        if enable:
            layer = self.display_data
            a = self._new_rgb_layer(layer)
            if a is None:
                return

            a.r = a.g = a.b = self.display_attribute

            with self.artists.ignore_empty():
                self.artists.pop(layer)
                self.artists.append(a)
            result = a
        else:
            with self.artists.ignore_empty():
                for artist in list(self.artists):
                    if isinstance(artist, RGBImageLayerBase):
                        self.artists.remove(artist)
                result = self.add_layer(layer)

        self._update_data_plot()
        self._redraw()
        return result

    def _update_aspect(self):
        self._update_data_plot(relim=True)
        self._redraw()

    def add_layer(self, layer):
        if layer in self.artists:
            return self.artists[layer][0]

        if layer.data not in self.data:
            raise TypeError("Data not managed by client's data collection")

        if not self.can_image_data(layer.data):
            # if data is 1D, try to scatter plot
            if len(layer.data.shape) == 1:
                return self.add_scatter_layer(layer)
            logging.getLogger(__name__).warning(
                "Cannot visualize %s. Aborting", layer.label)
            return

        if isinstance(layer, Data):
            result = self._new_image_layer(layer)
            self.artists.append(result)
            for s in layer.subsets:
                self.add_layer(s)
            self.set_data(layer)
        elif isinstance(layer, Subset):
            result = self._new_subset_image_layer(layer)
            self.artists.append(result)
            self._update_subset_single(layer)
        else:
            raise TypeError("Unrecognized layer type: %s" % type(layer))

        return result

    def add_scatter_layer(self, layer):
        logging.getLogger(
            __name__).debug('Adding scatter layer for %s' % layer)
        if layer in self.artists:
            logging.getLogger(__name__).debug('Layer already present')
            return

        result = self._new_scatter_layer(layer)
        self.artists.append(result)
        self._update_scatter_layer(layer)
        return result

    def _update_scatter_plots(self):
        for layer in self.artists.layers:
            self._update_scatter_layer(layer)

    @requires_data
    def _update_scatter_layer(self, layer, force=False):

        if layer not in self.artists:
            return

        xatt, yatt = self._get_plot_attributes()
        need_redraw = False

        for a in self.artists[layer]:
            if not isinstance(a, ScatterLayerBase):
                continue
            need_redraw = True
            a.xatt = xatt
            a.yatt = yatt
            if self.is_3D:
                zatt = self.display_data.get_pixel_component_id(
                    self._slice_ori)
                subset = (
                    zatt > self.slice_ind) & (zatt <= self.slice_ind + 1)
                a.emphasis = subset
            else:
                a.emphasis = None
            a.update() if not force else a.force_update()
            a.redraw()

        if need_redraw:
            self._redraw()

    @requires_data
    def _get_plot_attributes(self):
        x, y = _slice_axis(self.display_data.shape, self.slice)
        ids = self.display_data.pixel_component_ids
        return ids[x], ids[y]

    def _pixel_coords(self, x, y):
        """
        From a slice coordinate (x,y), return the full (possibly >2D) numpy
        index into the full data.

        .. note:: The inputs to this function are the reverse of numpy
                  convention (horizontal axis first, then vertical)

        Returns
        -------
        coords : tuple
            Either a tuple of (x,y) or (x,y,z)
        """
        result = list(self.slice)
        result[result.index('x')] = x
        result[result.index('y')] = y
        return result

    def is_visible(self, layer):
        return all(a.visible for a in self.artists[layer])

    def set_visible(self, layer, state):
        for a in self.artists[layer]:
            a.visible = state

    def set_slice_ori(self, ori):
        if not self.is_3D:
            raise IndexError("Can only set slice_ori for 3D images")
        if ori == 0:
            self.slice = (0, 'y', 'x')
        elif ori == 1:
            self.slice = ('y', 0, 'x')
        elif ori == 2:
            self.slice = ('y', 'x', 0)
        else:
            raise ValueError("Orientation must be 0, 1, or 2")

    def restore_layers(self, layers, context):
        """
        Restore a list of glue-serialized layer dicts.
        """
        for layer in layers:
            c = lookup_class_with_patches(layer.pop('_type'))
            props = dict((k, v if k == 'stretch' else context.object(v))
                         for k, v in layer.items())
            l = props['layer']
            if issubclass(c, ScatterLayerBase):
                l = self.add_scatter_layer(l)
            elif issubclass(c, RGBImageLayerBase):
                r = props.pop('r')
                g = props.pop('g')
                b = props.pop('b')
                self.display_data = l
                self.display_attribute = r
                l = self.rgb_mode(True)
                l.r = r
                l.g = g
                l.b = b
            elif issubclass(c, (ImageLayerBase, SubsetImageLayerBase)):
                if isinstance(l, Data):
                    self.set_data(l)
                l = self.add_layer(l)
            else:
                raise ValueError("Cannot restore layer of type %s" % l)
            l.properties = props

    def _on_component_replace(self, msg):
        if self.display_attribute is msg.old:
            self.display_attribute = msg.new

    def register_to_hub(self, hub):
        super(ImageClient, self).register_to_hub(hub)
        hub.subscribe(self,
                      ComponentReplacedMessage,
                      self._on_component_replace)

    # subclasses should override the following methods as appropriate
    def _new_rgb_layer(self, layer):
        """
        Construct and return an RGBImageLayerBase for the given layer

        Parameters
        ----------
        layer : :class:`~glue.core.data.Data` or :class:`~glue.core.subset.Subset`
            Which object to visualize
        """
        raise NotImplementedError()

    def _new_subset_image_layer(self, layer):
        """
        Construct and return a SubsetImageLayerArtist for the given layer

        Parameters
        ----------
        layer : :class:`~glue.core.data.Data` or :class:`~glue.core.subset.Subset`
            Which object to visualize
        """
        raise NotImplementedError()

    def _new_image_layer(self, layer):
        """
        Construct and return an ImageLayerArtist for the given layer

        Parameters
        ----------
        layer : :class:`~glue.core.data.Data` or :class:`~glue.core.subset.Subset`
            Which object to visualize
        """
        raise NotImplementedError()

    def _new_scatter_layer(self, layer):
        """
        Construct and return a ScatterLayerArtist for the given layer

        Parameters
        ----------
        layer : :class:`~glue.core.data.Data` or :class:`~glue.core.subset.Subset`
            Which object to visualize
        """
        raise NotImplementedError()

    def _update_axis_labels(self):
        """
        Sync the displays for labels on X/Y axes, because the data or slice has
        changed
        """
        raise NotImplementedError()

    def relim(self):
        """
        Reset view window to the default pan/zoom setting.
        """
        pass

    def show_crosshairs(self, x, y):
        pass

    def clear_crosshairs(self):
        pass


class MplImageClient(ImageClient):

    def __init__(self, data, figure=None, axes=None, layer_artist_container=None):
        super(MplImageClient, self).__init__(data, layer_artist_container)

        if axes is not None:
            raise ValueError("ImageClient does not accept an axes")
        self._setup_mpl(figure, axes)

        # description of field of view and center of image
        self._view_window = None

        # artist for a crosshair
        self._crosshairs = None

    def _setup_mpl(self, figure, axes):
        figure, axes = init_mpl(figure, axes, wcs=True)
        self._axes = axes
        self._axes.get_xaxis().set_ticks([])
        self._axes.get_yaxis().set_ticks([])
        self._figure = figure

        # custom axes formatter
        def format_coord(x, y):
            data = self.display_data
            if data is None:
                # MPL default method
                return type(self._axes).format_coord(self._axes, x, y)
            info = self.point_details(x, y)
            return '         '.join(info['labels'])

        self._axes.format_coord = format_coord

        self._cid = self._axes.figure.canvas.mpl_connect('button_release_event',
                                                         self.check_update)

        if hasattr(self._axes.figure.canvas, 'homeButton'):
            # test code doesn't always use Glue's custom FigureCanvas
            self._axes.figure.canvas.homeButton.connect(self.check_update)

    @property
    def axes(self):
        return self._axes

    def check_update(self, *args):
        """
        For the Matplotlib client, see if the view window has changed enough
        such that the images should be resampled
        """

        logging.getLogger(__name__).debug("check update")

        # We need to make sure we reapply the aspect ratio manually here,
        # because at this point, if the user has zoomed in to a region with a
        # different aspect ratio than the original view, Matplotlib has not yet
        # enforced computed the final limits. This is an issue if we have
        # requested square pixels.
        self.axes.apply_aspect()

        vw = _view_window(self._axes)

        if vw != self._view_window:
            logging.getLogger(__name__).debug("updating")
            self._update_and_redraw()
            self._view_window = vw

    def _update_and_redraw(self):

        self._update_data_plot()
        self._update_subset_plots()

        self._redraw()

    @requires_data
    def _update_axis_labels(self):
        labels = _axis_labels(self.display_data, self.slice)
        self._update_wcs_axes(self.display_data, self.slice)
        self._axes.set_xlabel(labels[1])
        self._axes.set_ylabel(labels[0])

    @defer_draw
    def _update_wcs_axes(self, data, slc):
        wcs = getattr(data.coords, 'wcs', None)

        if wcs is not None and hasattr(self.axes, 'reset_wcs'):
            self.axes.reset_wcs(wcs, slices=slc[::-1])

    def _redraw(self):
        self._axes.figure.canvas.draw()

    def relim(self):
        shp = _2d_shape(self.display_data.shape, self.slice)
        self._axes.set_xlim(0, shp[1])
        self._axes.set_ylim(0, shp[0])

    def _new_rgb_layer(self, layer):
        v = self._view or self._build_view()
        a = RGBImageLayerArtist(layer, self._axes, last_view=v)
        return a

    def _new_image_layer(self, layer):
        return ImageLayerArtist(layer, self._axes)

    def _new_subset_image_layer(self, layer):
        return SubsetImageLayerArtist(layer, self._axes)

    def _new_scatter_layer(self, layer):
        return ScatterLayerArtist(layer, self._axes)

    def _build_view(self):

        att = self.display_attribute
        shp = self.display_data.shape

        shp_2d = _2d_shape(shp, self.slice)
        v = extract_matched_slices(self._axes, shp_2d)
        x = slice(v[0], v[1], v[2])
        y = slice(v[3], v[4], v[5])

        slc = list(self.slice)
        slc[slc.index('x')] = x
        slc[slc.index('y')] = y
        return (att,) + tuple(slc)

    def show_crosshairs(self, x, y):
        if self._crosshairs is not None:
            self._crosshairs.remove()

        self._crosshairs, = self._axes.plot([x], [y], '+', ms=12,
                                            mfc='none', mec='#d32d26',
                                            mew=2, zorder=100)
        self._redraw()

    def clear_crosshairs(self):
        if self._crosshairs is not None:
            self._crosshairs.remove()
            self._crosshairs = None

    def register_to_hub(self, hub):

        super(MplImageClient, self).register_to_hub(hub)

        def is_appearance_settings(msg):
            return ('BACKGROUND_COLOR' in msg.settings
                    or 'FOREGROUND_COLOR' in msg.settings)

        hub.subscribe(self, SettingsChangeMessage,
                      self._update_appearance_from_settings,
                      filter=is_appearance_settings)

    def _update_appearance_from_settings(self, message):
        update_appearance_from_settings(self.axes)
        self._redraw()



def _2d_shape(shape, slc):
    """
    Return the shape of the 2D slice through a 2 or 3D image.
    """
    # - numpy ordering here
    return shape[slc.index('y')], shape[slc.index('x')]


def _slice_axis(shape, slc):
    """
    Return a 2-tuple of which axes in a dataset lie along the x and y axes of
    the image.

    Parameters
    ----------
    shape : tuple
        Shape of original data.
    slc : tuple
        Slice through the data, 'x', and 'y'
    """
    return slc.index('x'), slc.index('y')


def _axis_labels(data, slc):
    shape = data.shape
    names = [data.get_world_component_id(i).label
             for i in range(len(shape))]
    return names[slc.index('y')], names[slc.index('x')]


def _view_window(ax):
    """
    Return a tuple describing the view window of an axes object.

    The contents should not be used directly, Rather, several
    return values should be compared with == to determine if the
    window has been panned/zoomed
    """
    ext = (ax.transAxes.transform([(1, 1)]) - ax.transAxes.transform([(0, 0)]))[0]
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    result = xlim[0], ylim[0], xlim[1], ylim[1], ext[0], ext[1]
    logging.getLogger(__name__).debug("view window: %s", result)
    return result


def _default_component(data):
    """
    Choose a default ComponentID to display for data
    """
    cid = data.find_component_id('PRIMARY')
    if cid is not None:
        return cid
    return data.component_ids()[0]
