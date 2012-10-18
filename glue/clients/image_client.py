import logging

import numpy as np

from .modest_image import extract_matched_slices
from ..core.exceptions import IncompatibleAttribute
from ..core.data import Data
from ..core.subset import Subset, RoiSubsetState
from ..core.roi import PolygonalROI
from ..core.edit_subset_mode import EditSubsetMode

from .viz_client import VizClient, init_mpl
from .layer_artist import (ScatterLayerArtist, LayerArtistContainer,
                           ImageLayerArtist, SubsetImageLayerArtist)


def requires_data(func):
    """Decorator that checks an ImageClient for a non-null display_data
    attribute. Only executes decorated function if present"""
    def result(*args, **kwargs):
        if args[0].display_data is None:
            return
        return func(*args, **kwargs)
    return result


class ImageClient(VizClient):

    def __init__(self, data, figure=None, axes=None, artist_container=None):

        figure, axes = init_mpl(figure, axes)

        VizClient.__init__(self, data)

        self.artists = artist_container
        if self.artists is None:
            self.artists = LayerArtistContainer()

        self.display_data = None
        self.display_attribute = None
        self._slice_ori = 0
        self._slice_ind = 0
        self._view_window = None
        self._view = None
        self._image = None

        self._ax = axes
        self._ax.get_xaxis().set_ticks([])
        self._ax.get_yaxis().set_ticks([])
        self._figure = figure
        self._norm_cache = {}

        #format axes
        fc = self._ax.format_coord

        def format_coord(x, y):
            if self.display_data is None:
                return fc(x, y)
            pix = self._pixel_coords(x, y)
            world = self.display_data.coords.pixel2world(*pix)
            world = world[::-1]   # reverse for numpy convention
            ind = _slice_axis(self.display_data.shape, self._slice_ori)
            labels = _slice_labels(self.display_data, self._slice_ori)
            return '%s=%s          %s=%s' % (labels[1], world[ind[1]],
                                             labels[0], world[ind[0]])
        self._ax.format_coord = format_coord

        self._cid = self._ax.figure.canvas.mpl_connect('button_release_event',
                                                       self.check_update)

    @property
    def axes(self):
        return self._ax

    @property
    def is_3D(self):
        if not self.display_data:
            return False
        return len(self.display_data.shape) == 3

    @property
    def slice_ind(self):
        if self.is_3D:
            return self._slice_ind
        return None

    @property
    def image(self):
        return self._image

    @slice_ind.setter
    def slice_ind(self, value):
        if self.is_3D:
            self._slice_ind = value
            self._update_data_plot()
            self._update_subset_plots()
            self._redraw()
        else:
            raise IndexError("Cannot set slice for 2D image")

    def can_handle_data(self, data):
        return data.ndim in [2, 3]

    def _ensure_data_present(self, data):
        if data not in self.artists:
            self.add_layer(data)

    def check_update(self, event):
        logging.debug("check update")
        vw = _view_window(self._ax)
        if vw != self._view_window:
            logging.debug("updating")
            self._update_data_plot()
            self._update_subset_plots()
            self._redraw()
            self._view_window = vw

    def set_data(self, data, attribute=None):
        self._ensure_data_present(data)

        attribute = attribute or _default_component(data)

        self.display_data = data
        self.display_attribute = attribute
        self._update_axis_labels()
        self._update_data_plot(relim=True)
        self._update_subset_plots()
        self._redraw()

    def slice_bounds(self):
        if not self.is_3D:
            return (0, 0)
        if self._slice_ori == 2:
            return (0, self.display_data.shape[2] - 1)
        if self._slice_ori == 1:
            return (0, self.display_data.shape[1] - 1)
        if self._slice_ori == 0:
            return (0, self.display_data.shape[0] - 1)

    def set_slice_ori(self, ori):
        if not self.is_3D:
            raise IndexError("Cannot set orientation of 2D image")
        if ori not in [0, 1, 2]:
            raise TypeError("Orientation must be 0, 1, or 2")
        self._slice_ori = ori
        self.slice_ind = min(self.slice_ind, self.slice_bounds()[1])
        self.slice_ind = max(self.slice_ind, self.slice_bounds()[0])
        self._update_axis_labels()

        self._update_data_plot(relim=True)
        self._update_subset_plots()

        self._redraw()

    @requires_data
    def _update_axis_labels(self):
        ori = self._slice_ori
        labels = _slice_labels(self.display_data, ori)
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
            self.set_norm(*self._norm_cache[attribute])
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
    def set_norm(self, vmin, vmax):
        for a in self.artists[self.display_data]:
            a.set_norm(vmin, vmax)
        self._update_data_plot()
        self._redraw()

    @requires_data
    def clear_norm(self):
        for a in self.artists[self.display_data]:
            a.clear_norm()

    @requires_data
    def get_norm(self):
        a = self.artists[self.display_data][0]
        norm = a.norm
        return norm.vmin, norm.vmax

    @requires_data
    def set_cmap(self, cmap):
        for a in self.artists[self.display_data]:
            a.cmap = cmap
            a.redraw()

    def _build_view(self, matched=False):
        att = self.display_attribute
        shp = self.display_data.shape
        shp_2d = _2d_shape(shp, self._slice_ori)
        x, y = np.s_[:], np.s_[:]
        if matched:
            v = extract_matched_slices(self._ax, shp_2d)
            x = slice(v[0], v[1], v[2])
            y = slice(v[3], v[4], v[5])

        if not self.is_3D:
            return (att, y, x)
        if self._slice_ori == 0:
            return (att, self.slice_ind, y, x)
        if self._slice_ori == 1:
            return (att, y, self.slice_ind, x)
        assert self._slice_ori == 2
        return (att, y, x, self.slice_ind)

    @requires_data
    def _update_data_plot(self, relim=False):
        """
        Re-sync the main image and its subsets
        """

        if relim:
            self.relim()

        view = self._build_view(matched=True)
        self._image = self.display_data[view]

        self._view = view
        for a in list(self.artists):
            if (not isinstance(a, ScatterLayerArtist)) and \
                    a.layer.data is not self.display_data:
                self.artists.remove(a)
            else:
                a.update(view)
        for a in self.artists[self.display_data]:
            a.update(view)

    def relim(self):
        shp = _2d_shape(self.display_data.shape, self._slice_ori)
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
        logging.debug("update subset single: %s", s)
        self._update_scatter_layer(s)

        if s not in self.artists:
            return

        if s.data is not self.display_data:
            return

        view = self._build_view(matched=True)
        for a in self.artists[s]:
            a.update(view)

        if redraw:
            self._redraw()

    @requires_data
    def _apply_roi(self, roi):

        subset_state = RoiSubsetState()
        xroi, yroi = roi.to_polygon()
        x, y = self._get_axis_components()
        subset_state.xatt = x
        subset_state.yatt = y
        subset_state.roi = PolygonalROI(xroi, yroi)
        mode = EditSubsetMode()
        mode.update(self.data, subset_state, focus_data=self.display_data)

    def _horizontal_axis_index(self):
        """Which index (in numpy convention - zyx) does the horizontal
        axis coorespond to?"""
        if not self.is_3D or self._slice_ori == 2:
            return 1
        return 2

    def _vertical_axis_index(self):
        """Which index (in numpy convention - zyx) does the vertical
        axis coorespond to?"""
        if self.is_3D and self._slice_ori == 0:
            return 1
        return 0

    def _get_axis_components(self):
        data = self.display_data
        ids = [self._horizontal_axis_index(), self._vertical_axis_index()]
        return map(data.get_pixel_component_id, ids)

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
        self.add_layer(layer)

    def add_layer(self, layer):
        if layer in self.artists:
            return

        if layer.data not in self.data:
            raise TypeError("Data not managed by client's data collection")

        if not self.can_handle_data(layer.data):
            logging.warning("Cannot visulize %s. Aborting", layer.label)
            return

        if isinstance(layer, Data):
            self.artists.append(ImageLayerArtist(layer, self._ax))
            for s in layer.subsets:
                self.add_layer(s)
        elif isinstance(layer, Subset):
            self.artists.append(SubsetImageLayerArtist(layer, self._ax))
            self._update_subset_single(layer)
        else:
            raise TypeError("Unrecognized layer type: %s" % type(layer))

    def add_scatter_layer(self, layer):
        logging.getLogger(
            __name__).debug('Adding scatter layer for %s' % layer)
        if layer in self.artists:
            logging.getLogger(__name__).debug('Layer already present')
            return

        self.artists.append(ScatterLayerArtist(layer, self._ax))
        self._update_scatter_layer(layer)

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
                    zatt > self._slice_ind) & (zatt <= self._slice_ind + 1)
                a.emphasis = subset
            else:
                a.emphasis = None
            a.update()
            a.redraw()
        self._redraw()

    @requires_data
    def _get_plot_attributes(self):
        y, x = _slice_axis(self.display_data.shape, self._slice_ori)
        ids = self.display_data.pixel_component_ids
        return ids[x], ids[y]

    def _pixel_coords(self, x, y):
        """From a slice coordinate (x,y), return the full (possibly
        3D) location

        *Note*
        The order of inputs and outputs from this function are reverse
        the numpy convention (i.e. x axis specified first, not last)

        *Returns*
        Either (x,y) or (x,y,z)
        """
        if not self.is_3D:
            return x, y
        if self._slice_ori == 0:
            return x, y, self.slice_ind
        elif self._slice_ori == 1:
            return x, self.slice_ind, y
        else:
            assert self._slice_ori == 2
            return self.slice_ind, x, y


def _2d_shape(shape, slice_ori):
    """Return the shape of the 2D slice through a 2 or 3D image"""
    if len(shape) == 2:
        return shape
    if slice_ori == 0:
        return shape[1:]
    if slice_ori == 1:
        return shape[0], shape[2]
    assert slice_ori == 2
    return shape[0:2]


def _slice_axis(shape, slice_ori):
    """Return a 2-tuple of the axis indices for the given
    image and slice orientation"""
    if len(shape) == 2:
        return 0, 1
    if slice_ori == 0:
        return 1, 2
    if slice_ori == 1:
        return 0, 2
    assert slice_ori == 2
    return 0, 1


def _slice_labels(data, slice_ori):
    shape = data.shape
    names = [data.get_world_component_id(i).label
             for i in range(len(shape))]
    names = [n.split(':')[-1].split('-')[0] for n in names]
    if len(shape) == 2:
        return names[0], names[1]
    if slice_ori == 0:
        return names[1], names[2]
    if slice_ori == 1:
        return names[0], names[2]
    assert slice_ori == 2
    return names[0], names[1]


def _view_window(ax):
    """ Return a tuple describing the view window of an axes object.

    The contents should not be used directly, Rather, several
    return values should be compared with == to determine if the
    window has been panned/zoomed
    """
    ext = ax.transAxes.transform([1, 1]) - ax.transAxes.transform([0, 0])
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    result = xlim[0], ylim[0], xlim[1], ylim[1], ext[0], ext[1]
    logging.debug("view window: %s", result)
    return result


def _default_component(data):
    """Choose a default ComponentID to display for data

    Returns PRIMARY if present
    """
    cid = data.find_component_id('PRIMARY')
    if cid is not None:
        return cid
    return data.component_ids()[0]
