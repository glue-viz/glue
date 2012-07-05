import numpy as np
import matplotlib.pyplot as plt

from ..core.client import Client
from ..core.exceptions import IncompatibleAttribute
from ..core.data import Data
from ..core.subset import RoiSubsetState
from ..core.roi import PolygonalROI
from ..core.util import relim


class ScatterLayerManager(object):

    def __init__(self, layer, axes):
        self._layer = layer
        self._axes = axes
        self._visible = True
        self._enabled = True
        self._artist = None
        self._init_artist()

    def _init_artist(self):
        artist = self._axes.scatter([1], [1])
        artist.set_offsets(np.zeros((0, 2)))
        self._artist = artist

    def _remove_artist(self):
        if self._artist in self._axes.collections:
            self._artist.remove()

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
        return self._artist.get_offsets()

    def sync_style(self):
        style = self._layer.style
        artist = self._artist
        artist.set_edgecolor('none')
        artist.set_facecolor(style.color)
        try:
            artist.get_sizes().data[0] = style.markersize
        except TypeError:
            artist.get_sizes()[0] = style.markersize

        artist.set_alpha(style.alpha)

    def set_data(self, x, y):
        xy = np.zeros((x.size, 2))
        xy[:, 0] = x.flat
        xy[:, 1] = y.flat
        self._artist.set_offsets(xy)

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
    def __init__(self, data=None, figure=None, axes=None):
        """
        Create a new ScatterClient object

        Inputs:
        =======
        data : `glue.data.DataCollection` instance

        figure : matplotlib Figure instance (optional)
           Which figure instance to draw to. One will be created if
           not provided

        axes : matplotlib Axes instance (optional)
           Which axes instance to use. Will be created if necessary
        """
        Client.__init__(self, data=data)

        self.managers = {}

        self._xatt = None
        self._yatt = None
        self._layer_updated = False  # debugging

        if figure is None:
            if axes is not None:
                figure = axes.figure
            else:
                figure = plt.figure()
        if axes is None:
            ax = figure.add_subplot(111)
        else:
            if axes.figure is not figure:
                raise TypeError("Axes and Figure inputs do not "
                                "belong to each other")
            ax = axes
        self.ax = ax

    def is_layer_present(self, layer):
        """ True if layer is plotted """
        return layer in self.managers

    def get_layer_order(self, layer):
        return self.managers[layer].get_zorder()

    def plottable_attributes(self, layer):
        data = layer.data
        return [c for c in data.component_ids() if
                np.can_cast(data[c].dtype, np.float)]

    def add_layer(self, layer):
        """ Adds a new visual layer to a client, to display either a dataset
        or a subset. Updates both the client data structure and the
        plot.

        Inputs:
        =======
        layer : `glue.data.Data` or `glue.subset.Subset` object
            The layer to add
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

    def _snap_xlim(self):
        """
        Reset the plotted x rng to show all the data
        """
        rng = [np.infty, -np.infty]
        is_log = self.ax.get_xscale() == 'log'
        for layer in self.managers:
            manager = self.managers[layer]
            if not manager.is_visible():
                continue
            xy = manager.get_data()
            if xy.shape[0] == 0:
                continue
            rng0 = relim(min(xy[:, 0]), max(xy[:, 0]), is_log)
            rng[0] = min(rng[0], rng0[0])
            rng[1] = max(rng[1], rng0[1])
        if rng[0] == np.infty:
            return

        if self.ax.xaxis_inverted():
            rng = [rng[1], rng[0]]

        self.ax.set_xlim(rng)

    def _snap_ylim(self):
        """
        Reset the plotted y rng to show all the data
        """
        rng = [np.infty, -np.infty]
        is_log = self.ax.get_yscale() == 'log'
        for layer in self.managers:
            manager = self.managers[layer]
            if not manager.is_visible():
                continue
            xy = manager.get_data()
            if xy.shape[0] == 0:
                continue
            rng0 = relim(min(xy[:, 1]), max(xy[:, 1]), is_log)
            rng[0] = min(rng[0], rng0[0])
            rng[1] = max(rng[1], rng0[1])
        if rng[0] == np.infty:
            return

        if self.ax.xaxis_inverted():
            rng = [rng[1], rng[0]]

        self.ax.set_ylim(rng)

    def set_visible(self, layer, state):
        """ Toggle a layer's visibility

        Inputs:
        =======
        layer : `glue.data.Data` or `glue.subset.Subset` instance
              Which layer to modify
        state : boolean
              True to show. False to hide
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

        Inputs:
        =======
        coord : 'x' or 'y'
           Which axis to reassign
        attribute : string
           Which attribute of the data to use.
        snap : bool
           If True, will rescale x/y axes to fit the data
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

    def _apply_roi(self, roi):
        # every active data layer is set
        # using specified ROI
        for layer in self.managers:
            if layer.data is not layer:
                continue
            if not self.managers[layer].is_enabled():
                continue
            subset_state = RoiSubsetState()
            subset_state.xatt = self._xatt
            subset_state.yatt = self._yatt
            x, y = roi.to_polygon()
            subset_state.roi = PolygonalROI(x, y)
            subset = layer.edit_subset
            subset.subset_state = subset_state

    def set_xdata(self, attribute, snap=True):
        """
        Redefine which component gets plotted on the x axis

        Parameters
        ----------
        attribute : string
                 The name of the new data component to plot
        snap : bool
             If true, re-scale x axis to show all values
        """
        self.set_xydata('x', attribute, snap=snap)

    def set_ydata(self, attribute, snap=True):
        """
        Redefine which component gets plotted on the y axis

        Parameters
        ----------
        attribute: string
                  The name of the new data component to plot

        snap : bool
               If True, re-scale y axis to show all values
        """
        self.set_xydata('y', attribute, snap=snap)

    def set_xlog(self, state):
        """ Set the x axis scaling

        Inputs:
        =======
        state : string ('log' or 'linear')
            The new scaling for the x axis
        """
        mode = 'log' if state else 'linear'
        self.ax.set_xscale(mode)
        self._redraw()

    def set_ylog(self, state):
        """ Set the y axis scaling

        Inputs:
        =======
        state : string ('log' or 'linear')
            The new scaling for the y axis
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

        Inputs:
        =======
        state : bool
            True to flip x axis
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
