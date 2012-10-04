import logging

import numpy as np
import matplotlib.pyplot as plt

from ..core.message import DataCollectionAddMessage
from ..core.client import Client
from ..core.exceptions import IncompatibleAttribute
from ..core.data import Data
from ..core.subset import RoiSubsetState
from ..core.roi import PolygonalROI
from ..core.util import relim
from ..core.edit_subset_mode import EditSubsetMode
from .viz_client import init_mpl


class ScatterLayerManager(object):

    def __init__(self, layer, axes):
        self._layer = layer
        self._axes = axes
        self._visible = True
        self._enabled = True
        self._artist = None
        self._init_artist()
        self._x = np.array([])
        self._y = np.array([])

    def _init_artist(self):
        artist, = self._axes.plot([1], [1])
        artist.set_data(np.array([np.nan]), np.array([np.nan]))
        self._artist = artist

    def _remove_artist(self):
        try:
            self._artist.remove()
        except ValueError:  # already removed
            pass

    def set_enabled(self, state):
        self._enabled = state
        self._artist.set_visible(state and self.is_visible())

    def set_visible(self, state):
        self._visible = state
        self._artist.set_visible(state and self.is_enabled())

    def is_enabled(self):
        return self._enabled

    def is_visible(self):
        return self._visible

    def get_data(self):
        result = self._x, self._y
        logging.getLogger(__name__).debug("get data result: %s",
                                          result)
        return result

    def sync_style(self):
        style = self._layer.style
        artist = self._artist
        artist.set_markeredgecolor('none')
        artist.set_markerfacecolor(style.color)
        artist.set_marker(style.marker)
        artist.set_markersize(style.markersize)
        artist.set_linestyle('None')
        artist.set_alpha(style.alpha)

    def set_data(self, x, y):
        self._x = x.ravel()
        self._y = y.ravel()
        self._artist.set_data(self._x, self._y)

    def set_zorder(self, order):
        self._artist.set_zorder(order)

    def get_zorder(self):
        return self._artist.get_zorder()

    def __del__(self):
        self._remove_artist()


class ScatterClient(Client):
    """
    A client class that uses matplotlib to visualize tables as scatter plots.
    """
    def __init__(self, data=None, figure=None, axes=None, master_data=None):
        """
        Create a new ScatterClient object

        :param data: :class:`~glue.core.DataCollection` to use

        :param figure:
           Which matplotlib figure instance to draw to. One will be created if
           not provided

        :param axes:
           Which matplotlib axes instance to use. Will be created if necessary

        :param master_data:
           An optional superset of the data, if this client is to show a
           subset of the data. XXX Refactor this ugliness
        """
        Client.__init__(self, data=data)
        figure, axes = init_mpl(figure, axes)
        self._master_data = master_data or data
        self.managers = {}

        self._xatt = None
        self._yatt = None
        self._layer_updated = False  # debugging

        self.ax = axes

    @property
    def axes(self):
        return self.ax

    def register_to_hub(self, hub):
        super(ScatterClient, self).register_to_hub(hub)
        data_in_dc = lambda x: x.data in self._data
        hub.subscribe(self,
                      DataCollectionAddMessage,
                      handler=lambda x: self.add_layer(x.data),
                      filter=data_in_dc)

    def is_layer_present(self, layer):
        """ True if layer is plotted """
        return layer in self.managers

    def get_layer_order(self, layer):
        return self.managers[layer].get_zorder()

    def plottable_attributes(self, layer):
        data = layer.data
        return [c for c in data.visible_components if
                np.can_cast(data[c].dtype, np.float)]

    def add_layer(self, layer):
        """ Adds a new visual layer to a client, to display either a dataset
        or a subset. Updates both the client data structure and the
        plot.

        :param layer: the layer to add
        :type layer: :class:`~glue.core.Data` or :class:`~glue.core.Subset`
        """
        if layer.data not in self.data:
            raise TypeError("Layer not in data collection")
        if layer in self.managers:
            return
        self.managers[layer] = ScatterLayerManager(layer, self.ax)
        self._bring_subsets_to_front()
        self._update_layer(layer)
        self._ensure_subsets_added(layer)

    def _ensure_subsets_added(self, layer):
        if not isinstance(layer, Data):
            return
        for subset in layer.subsets:
            self.add_layer(subset)

    def _bring_subsets_to_front(self):
        """ Make sure subsets are in front of data """
        nlayers = len(self.managers)
        for i, data in enumerate(self.data):
            if data not in self.managers:
                continue
            self.managers[data].set_zorder(i * nlayers)
            for j, sub in enumerate(data.subsets):
                if sub not in self.managers:
                    continue
                self.managers[sub].set_zorder(i * nlayers + j + 1)

    def _visible_limits(self, axis):
        """Return the min-max visible data boundaries for given axis"""
        data = []
        for mgr in self.managers.values():
            if not mgr.is_visible():
                continue
            xy = mgr.get_data()
            assert isinstance(xy, tuple)
            data.append(xy[axis])

        if len(data) == 0:
            return
        data = np.hstack(data)
        if data.size == 0:
            return

        lo, hi = np.nanmin(data), np.nanmax(data)
        if not np.isfinite(lo):
            return

        return lo, hi

    def _snap_xlim(self):
        """
        Reset the plotted x rng to show all the data
        """
        is_log = self.ax.get_xscale() == 'log'
        rng = self._visible_limits(0)
        if rng is None:
            return
        rng = relim(rng[0], rng[1], is_log)

        if self.ax.xaxis_inverted():
            rng = rng[::-1]

        self.ax.set_xlim(rng)

    def _snap_ylim(self):
        """
        Reset the plotted y rng to show all the data
        """
        rng = [np.infty, -np.infty]
        is_log = self.ax.get_yscale() == 'log'

        rng = self._visible_limits(1)
        if rng is None:
            return
        rng = relim(rng[0], rng[1], is_log)

        if self.ax.yaxis_inverted():
            rng = rng[::-1]

        self.ax.set_ylim(rng)

    def snap(self):
        """Rescale axes to fit the data"""
        self._snap_xlim()
        self._snap_ylim()
        self._redraw()

    def set_visible(self, layer, state):
        """ Toggle a layer's visibility

        :param layer: which layer to modify
        :type layer: class:`~glue.core.Data` or :class:`~glue.coret.Subset`

        :param state: True to show. false to hide
        :type state: boolean
        """
        if layer not in self.managers:
            return
        self.managers[layer].set_visible(state)
        self._redraw()

    def is_visible(self, layer):
        if layer not in self.managers:
            return False
        return self.managers[layer].is_visible()

    def set_xydata(self, coord, attribute, snap=True):
        """ Redefine which components get assigned to the x/y axes

        :param coord: 'x' or 'y'
           Which axis to reassign
        :param attribute:
           Which attribute of the data to use.
        :type attribute: str
        :param snap:
           If True, will rescale x/y axes to fit the data
        :type snap: bool
        """

        if coord not in ('x', 'y'):
            raise TypeError("coord must be one of x,y")

        #update coordinates of data and subsets
        if coord == 'x':
            self._xatt = attribute
        elif coord == 'y':
            self._yatt = attribute

        #update plots
        map(self._update_layer, (l for l in self.managers))

        if coord == 'x' and snap:
            self._snap_xlim()
        elif coord == 'y' and snap:
            self._snap_ylim()

        self._update_axis_labels()
        self._redraw()

    def _apply_roi(self, roi):
        # every editable subset is updated
        # using specified ROI
        subset_state = RoiSubsetState()
        subset_state.xatt = self._xatt
        subset_state.yatt = self._yatt
        x, y = roi.to_polygon()
        subset_state.roi = PolygonalROI(x, y)
        mode = EditSubsetMode()
        for d in self._master_data:
            focus = d if self.is_visible(d) else None
            mode.update(d, subset_state, focus_data=focus)

    def set_xdata(self, attribute, snap=True):
        """
        Redefine which component gets plotted on the x axis

        :param attribute:
                 The name of the new data component to plot
        :type attribute: str
        :param snap:
             If true, re-scale x axis to show all values
        :type snap: bool
        """
        self.set_xydata('x', attribute, snap=snap)

    def set_ydata(self, attribute, snap=True):
        """
        Redefine which component gets plotted on the y axis

        :param attribute:
           The name of the new data component to plot
        :type attribute: string

        :param snap:
               If True, re-scale y axis to show all values
        :type snap: bool
        """
        self.set_xydata('y', attribute, snap=snap)

    def set_xlog(self, state):
        """ Set the x axis scaling

        :param state:
            The new scaling for the x axis
        :type state: string ('log' or 'linear')
        """
        mode = 'log' if state else 'linear'
        self.ax.set_xscale(mode)
        self._redraw()

    def set_ylog(self, state):
        """ Set the y axis scaling

        :param state: The new scaling for the y axis
        :type state: string ('log' or 'linear')
        """
        mode = 'log' if state else 'linear'
        self.ax.set_yscale(mode)
        self._redraw()

    def is_xflip(self):
        xlim = self.ax.get_xlim()
        return xlim[1] <= xlim[0]

    def is_yflip(self):
        ylim = self.ax.get_ylim()
        return ylim[1] <= ylim[0]

    def is_xlog(self):
        return self.ax.get_xscale() == 'log'

    def is_ylog(self):
        return self.ax.get_yscale() == 'log'

    def set_xflip(self, state):
        """ Set whether the x axis increases or decreases to the right.

        :param state: True to flip x axis

        """
        rng = self.ax.get_xlim()
        if state:
            self.ax.set_xlim(max(rng), min(rng))
        else:
            self.ax.set_xlim(min(rng), max(rng))
        self._redraw()

    def set_yflip(self, state):
        rng = self.ax.set_ylim()
        if state:
            self.ax.set_ylim(max(rng), min(rng))
        else:
            self.ax.set_ylim(min(rng), max(rng))
        self._redraw()

    def _remove_layer_artists(self, layer):
        if layer not in self.managers:
            return
        self.managers[layer].remove_artist()

    def _remove_data(self, message):
        for s in message.data.subsets:
            self.delete_layer(s)
        self.delete_layer(message.data)

    def _remove_subset(self, message):
        self.delete_layer(message.subset)

    def delete_layer(self, layer):
        if layer not in self.managers:
            return
        manager = self.managers.pop(layer)
        del manager
        self._redraw()
        assert not self.is_layer_present(layer)

    def _update_data(self, message):
        data = message.sender
        self._update_layer(data)

    def _redraw(self):
        self.ax.figure.canvas.draw()

    def _update_axis_labels(self):
        self.ax.set_xlabel(self._xatt)
        self.ax.set_ylabel(self._yatt)

    def _add_subset(self, message):
        subset = message.sender
        subset.do_broadcast(False)
        self.add_layer(subset)
        subset.do_broadcast(True)

    def add_data(self, data):
        self.add_layer(data)
        for subset in data.subsets:
            self.add_layer(subset)

    @property
    def data(self):
        """The data objects in the scatter plot"""
        return list(self._data)

    def _update_subset(self, message):
        self._update_layer(message.sender)

    def _update_layer(self, layer):
        """ Update both the style and data for the requested layer"""
        if self._xatt is None or self._yatt is None:
            return

        if layer not in self.managers:
            return

        try:
            x = layer[self._xatt]
            y = layer[self._yatt]
        except IncompatibleAttribute:
            self.managers[layer].set_enabled(False)
            self._redraw()
            return

        self._layer_updated = True

        self.managers[layer].set_enabled(True)
        self.managers[layer].set_data(x, y)
        self.managers[layer].sync_style()
        self._redraw()
