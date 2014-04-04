import logging
from functools import wraps

import numpy as np

from .modest_image import extract_matched_slices
from ..core.exceptions import IncompatibleAttribute
from ..core.data import Data
from ..core.util import lookup_class
from ..core.subset import Subset, RoiSubsetState
from ..core.roi import PolygonalROI
from ..core.callback_property import (
    callback_property, CallbackProperty)
from ..core.edit_subset_mode import EditSubsetMode

from .viz_client import VizClient, init_mpl
from .layer_artist import (ScatterLayerArtist, LayerArtistContainer,
                           ImageLayerArtist, SubsetImageLayerArtist,
                           RGBImageLayerArtist)


def requires_data(func):
    """Decorator that checks an ImageClient for a non-null display_data
    attribute. Only executes decorated function if present"""
    @wraps(func)
    def result(*args, **kwargs):
        if args[0].display_data is None:
            return
        return func(*args, **kwargs)
    return result


class ImageClient(VizClient):
    display_data = CallbackProperty(None)
    display_attribute = CallbackProperty(None)

    def __init__(self, data, figure=None, axes=None, artist_container=None):

        figure, axes = init_mpl(figure, axes)

        VizClient.__init__(self, data)

        self.artists = artist_container
        if self.artists is None:
            self.artists = LayerArtistContainer()

        self._slice = None
        self._view_window = None
        self._view = None
        self._image = None
        self._override_image = None

        self._ax = axes
        self._ax.get_xaxis().set_ticks([])
        self._ax.get_yaxis().set_ticks([])
        self._figure = figure
        self._norm_cache = {}

        # format axes
        fc = self._ax.format_coord

        def format_coord(x, y):
            data = self.display_data
            if data is None:
                return fc(x, y)
            info = self.point_details(x, y)
            return '         '.join(info['labels'])

        self._ax.format_coord = format_coord

        self._cid = self._ax.figure.canvas.mpl_connect('button_release_event',
                                                       self.check_update)
        if hasattr(self._ax.figure.canvas, 'homeButton'):
            # test code doesn't always use Glue's custom FigureCanvas
            self._ax.figure.canvas.homeButton.connect(self.check_update)

    def point_details(self, x, y):
        data = self.display_data
        pix = self._pixel_coords(x, y)
        world = data.coords.pixel2world(*pix[::-1])
        world = world[::-1]   # reverse for numpy convention
        labels = ['%s=%s' % (data.get_world_component_id(i).label, w)
                  for i, w in enumerate(world)]

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

    @callback_property
    def slice(self):
        """
        Returns a tuple describing the current slice through the data

        The tuple has length equal to the dimensionality of the display
        data. Each entry is either:

        'x' if the dimension is mapped to the X image axis
        'y' if the dimension is mapped to the Y image axis
        a number, indicating which fixed slice the dimension is restricted to
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
    def slice(self, value):
        if self.slice == tuple(value):
            return

        relim = value.index('x') != self._slice.index('x') or \
            value.index('y') != self._slice.index('y')
        self._slice = tuple(value)
        self._clear_override()
        self._update_axis_labels()
        self._update_data_plot(relim=relim)
        self._update_subset_plots()
        self._redraw()

    @property
    def axes(self):
        return self._ax

    @property
    def is_3D(self):
        """
        Returns True if the display data has 3 dimensions """
        if not self.display_data:
            return False
        return len(self.display_data.shape) == 3

    @property
    def slice_ind(self):
        """
        For 3D data, returns the pixel index of the current slice.
        Otherwise, returns None
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
        """Temporarily override the current slice view with another
        image (i.e., an aggregate)
        """
        self._override_image = image
        for a in self.artists[self.display_data]:
            if isinstance(a, ImageLayerArtist):
                a.override_image(image)
        self._update_data_plot()
        self._redraw()

    def _clear_override(self):
        self._override_image = None
        for a in self.artists[self.display_data]:
            if isinstance(a, ImageLayerArtist):
                a.clear_override()

    @slice_ind.setter
    def slice_ind(self, value):
        if self.is_3D:
            slc = [s if s in ['x', 'y'] else value for s in self.slice]
            self.slice = slc
            self._update_data_plot()
            self._update_subset_plots()
            self._redraw()
        else:
            raise IndexError("Can only set slice_ind for 3D images")

    def can_image_data(self, data):
        return data.ndim > 1

    def _ensure_data_present(self, data):
        if data not in self.artists:
            self.add_layer(data)

    def check_update(self, *args):
        logging.getLogger(__name__).debug("check update")
        vw = _view_window(self._ax)
        if vw != self._view_window:
            logging.getLogger(__name__).debug("updating")
            self._update_data_plot()
            self._update_subset_plots()
            self._redraw()
            self._view_window = vw

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
        self._redraw()

    @requires_data
    def _update_axis_labels(self):
        labels = _axis_labels(self.display_data, self.slice)
        self._ax.set_xlabel(labels[1])
        self._ax.set_ylabel(labels[0])

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
        Re-render the screen
        """
        self._ax.figure.canvas.draw()

    @requires_data
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
    def set_cmap(self, cmap):
        for a in self.artists[self.display_data]:
            a.cmap = cmap
            a.redraw()

    def _build_view(self, matched=False):
        att = self.display_attribute
        shp = self.display_data.shape
        shp_2d = _2d_shape(shp, self.slice)
        x, y = np.s_[:], np.s_[:]
        if matched:
            v = extract_matched_slices(self._ax, shp_2d)
            x = slice(v[0], v[1], v[2])
            y = slice(v[3], v[4], v[5])

        slc = list(self.slice)
        slc[slc.index('x')] = x
        slc[slc.index('y')] = y
        return (att,) + tuple(slc)

    @requires_data
    def _update_data_plot(self, relim=False):
        """
        Re-sync the main image and its subsets
        """

        if relim:
            self.relim()

        view = self._build_view(matched=True)
        self._image = self.display_data[view]
        transpose = self.slice.index('x') < self.slice.index('y')

        self._view = view
        for a in list(self.artists):
            if (not isinstance(a, ScatterLayerArtist)) and \
                    a.layer.data is not self.display_data:
                self.artists.remove(a)
            else:
                a.update(view, transpose)
        for a in self.artists[self.display_data]:
            a.update(view, transpose=transpose)

    def relim(self):
        shp = _2d_shape(self.display_data.shape, self.slice)
        self._ax.set_xlim(0, shp[1])
        self._ax.set_ylim(0, shp[0])

    def _update_subset_single(self, s, redraw=False):
        """
        Update the location and visual properties
        of each point in a single subset

        Parameters:
        ----------
        s: A subset instance
        The subset to refresh.

        """
        logging.getLogger(__name__).debug("update subset single: %s", s)
        self._update_scatter_layer(s)

        if s not in self.artists:
            return

        if s.data is not self.display_data:
            return

        view = self._build_view(matched=True)
        transpose = self.slice.index('x') < self.slice.index('y')
        for a in self.artists[s]:
            a.update(view, transpose)

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

        if layer is self.display_data:
            self.display_data = None

        if isinstance(layer, Data):
            for subset in layer.subsets:
                self.delete_layer(subset)

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
        """ Query whether RGB mode is enabled, or toggle RGB mode

        :param enable: bool, or None
        If True or False, explicitly enable/disable RGB mode.
        If None, check if RGB mode is enabled

        :rtype: LayerArtist or None
          If RGB mode is enabled, returns an RGBImageLayerArtist
          If enable=False, return the new ImageLayerArtist
        """
        # XXX need to better handle case where two RGBImageLayerArtists
        #    are created

        if enable is None:
            for a in self.artists:
                if isinstance(a, RGBImageLayerArtist):
                    return a
            return None

        result = None
        layer = self.display_data
        if enable:
            layer = self.display_data
            v = self._view or self._build_view(matched=True)
            a = RGBImageLayerArtist(layer, self._ax, last_view=v)

            for artist in self.artists.pop(layer):
                artist.clear()
            self.artists.append(a)
            result = a
        else:
            for artist in list(self.artists):
                if isinstance(artist, RGBImageLayerArtist):
                    artist.clear()
                self.artists.remove(artist)
            result = self.add_layer(layer)

        self._update_data_plot()
        self._redraw()
        return result

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
            result = ImageLayerArtist(layer, self._ax)
            self.artists.append(result)
            for s in layer.subsets:
                self.add_layer(s)
        elif isinstance(layer, Subset):
            result = SubsetImageLayerArtist(layer, self._ax)
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

        result = ScatterLayerArtist(layer, self._ax)
        self.artists.append(result)
        self._update_scatter_layer(layer)
        return result

    @requires_data
    def _update_scatter_layer(self, layer):
        xatt, yatt = self._get_plot_attributes()
        for a in self.artists[layer]:
            if not isinstance(a, ScatterLayerArtist):
                continue
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
            a.update()
            a.redraw()
        self._redraw()

    @requires_data
    def _get_plot_attributes(self):
        x, y = _slice_axis(self.display_data.shape, self.slice)
        ids = self.display_data.pixel_component_ids
        return ids[x], ids[y]

    def _pixel_coords(self, x, y):
        """From a slice coordinate (x,y), return the full (possibly
        >2D) numpy index into the full data

        *Note*
        The inputs to this function are the reverse of numpy convention
        (horizontal axis first, then vertical)


        *Returns*
        Either (x,y) or (x,y,z)
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
        """ Restore a list of glue-serialized layer dicts """
        for layer in layers:
            c = lookup_class(layer.pop('_type'))
            props = dict((k, v if k == 'stretch' else context.object(v))
                         for k, v in layer.items())
            l = props['layer']
            if c == ScatterLayerArtist:
                l = self.add_scatter_layer(l)
            elif c == ImageLayerArtist or c == SubsetImageLayerArtist:
                if isinstance(l, Data):
                    self.set_data(l)
                l = self.add_layer(l)
            elif c == RGBImageLayerArtist:
                r = props.pop('r')
                g = props.pop('g')
                b = props.pop('b')
                self.display_data = l
                self.display_attribute = r
                l = self.rgb_mode(True)
                l.r = r
                l.g = g
                l.b = b
            else:
                raise ValueError("Cannot restore layer of type %s" % l)
            l.properties = props


def _2d_shape(shape, slc):
    """Return the shape of the 2D slice through a 2 or 3D image
    """
    # - numpy ordering here
    return shape[slc.index('y')], shape[slc.index('x')]


def _slice_axis(shape, slc):
    """
    Return a 2-tuple of which axes in a dataset lie along the
    x and y axes of the image

    :param shape: Shape of original data. tuple of ints
    :param slc: Slice through the data, tuple of ints, 'x', and 'y'
    """
    return slc.index('x'), slc.index('y')


def _axis_labels(data, slc):
    shape = data.shape
    names = [data.get_world_component_id(i).label
             for i in range(len(shape))]
    return names[slc.index('y')], names[slc.index('x')]


def _view_window(ax):
    """ Return a tuple describing the view window of an axes object.

    The contents should not be used directly, Rather, several
    return values should be compared with == to determine if the
    window has been panned/zoomed
    """
    ext = ax.transAxes.transform([1, 1]) - ax.transAxes.transform([0, 0])
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    result = xlim[0], ylim[0], xlim[1], ylim[1], ext[0], ext[1]
    logging.getLogger(__name__).debug("view window: %s", result)
    return result


def _default_component(data):
    """Choose a default ComponentID to display for data

    Returns PRIMARY if present
    """
    cid = data.find_component_id('PRIMARY')
    if cid is not None:
        return cid
    return data.component_ids()[0]
