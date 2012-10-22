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
from .layer_artist import ScatterLayerArtist, LayerArtistContainer
from .util import visible_limits


class ScatterClient(Client):
    """
    A client class that uses matplotlib to visualize tables as scatter plots.
    """
    def __init__(self, data=None, figure=None, axes=None,
                 artist_container=None):
        """
        Create a new ScatterClient object

        :param data: :class:`~glue.core.DataCollection` to use

        :param figure:
           Which matplotlib figure instance to draw to. One will be created if
           not provided

        :param axes:
           Which matplotlib axes instance to use. Will be created if necessary
        """
        Client.__init__(self, data=data)
        figure, axes = init_mpl(figure, axes)
        self.artists = artist_container
        if self.artists is None:
            self.artists = LayerArtistContainer()

        self._xatt = None
        self._yatt = None
        self._layer_updated = False  # debugging

        self.ax = axes

    @property
    def axes(self):
        return self.ax

    def is_layer_present(self, layer):
        """ True if layer is plotted """
        return layer in self.artists

    def get_layer_order(self, layer):
        """If layer exists as a single artist, return its zorder.
        Otherwise, return None"""
        artists = self.artists[layer]
        if len(artists) == 1:
            return artists[0].zorder
        else:
            return None

    @property
    def layer_count(self):
        return len(self.artists)

    def plottable_attributes(self, layer, show_hidden=False):
        data = layer.data
        comp = data.components if show_hidden else data.visible_components
        return [c for c in comp if
                np.can_cast(data.dtype(c), np.float)]

    def add_layer(self, layer):
        """ Adds a new visual layer to a client, to display either a dataset
        or a subset. Updates both the client data structure and the
        plot.

        :param layer: the layer to add
        :type layer: :class:`~glue.core.Data` or :class:`~glue.core.Subset`
        """
        if layer.data not in self.data:
            raise TypeError("Layer not in data collection")
        if layer in self.artists:
            return
        self.artists.append(ScatterLayerArtist(layer, self.ax))
        self._update_layer(layer)
        self._ensure_subsets_added(layer)

    def _ensure_subsets_added(self, layer):
        if not isinstance(layer, Data):
            return
        for subset in layer.subsets:
            self.add_layer(subset)

    def _bring_subsets_to_front(self):
        """ Make sure subsets are in front of data """
        #XXX is this needed?
        nlayers = len(self.artists)
        for i, data in enumerate(self.data):
            if data not in self.artists:
                continue
            for a in self.artists[data]:
                a.zorder = i * nlayers
            for j, sub in enumerate(data.subsets):
                for a in self.artists[sub]:
                    a.zorder = i * nlayers + j + 1

    def _visible_limits(self, axis):
        """Return the min-max visible data boundaries for given axis"""
        return visible_limits(self.artists, axis)

    def _snap_xlim(self):
        """
        Reset the plotted x rng to show all the data
        """
        is_log = self.ax.get_xscale() == 'log'
        rng = self._visible_limits(0)
        if rng is None:
            return
        rng = relim(rng[0], rng[1], is_log)

        if self.is_xflip():
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

        if self.is_yflip():
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
        if layer not in self.artists:
            return
        for a in self.artists[layer]:
            a.visible = state
        self._redraw()

    def is_visible(self, layer):
        if layer not in self.artists:
            return False
        return any(a.visible for a in self.artists[layer])

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
        map(self._update_layer, self.artists.layers)

        if coord == 'x' and snap:
            self._snap_xlim()
        elif coord == 'y' and snap:
            self._snap_ylim()

        self._update_axis_labels()
        self._redraw()

    def apply_roi(self, roi):
        # every editable subset is updated
        # using specified ROI
        subset_state = RoiSubsetState()
        subset_state.xatt = self._xatt
        subset_state.yatt = self._yatt
        x, y = roi.to_polygon()
        subset_state.roi = PolygonalROI(x, y)
        mode = EditSubsetMode()
        visible = [d for d in self._data if self.is_visible(d)]
        focus = visible[0] if len(visible) > 0 else None
        mode.update(self._data, subset_state, focus_data=focus)

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
        lim = self.ax.get_xlim()
        self.ax.set_xscale(mode)

        #Rescale if switching to log with negative bounds
        if state and min(lim) <= 0:
            self._snap_xlim()

        self._redraw()

    def set_ylog(self, state):
        """ Set the y axis scaling

        :param state: The new scaling for the y axis
        :type state: string ('log' or 'linear')
        """
        mode = 'log' if state else 'linear'
        lim = self.ax.get_ylim()
        self.ax.set_yscale(mode)
        #Rescale if switching to log with negative bounds
        if state and min(lim) <= 0:
            self._snap_ylim()

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

    def _remove_data(self, message):
        """Process DataCollectionDeleteMessage"""
        for s in message.data.subsets:
            self.delete_layer(s)
        self.delete_layer(message.data)

    def _remove_subset(self, message):
        self.delete_layer(message.subset)

    def delete_layer(self, layer):
        if layer not in self.artists:
            return
        self.artists.pop(layer)
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
        #only add subset if data layer present
        if subset.data not in self.artists:
            return
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

        if layer not in self.artists:
            return

        self._layer_updated = True
        for art in self.artists[layer]:
            art.xatt = self._xatt
            art.yatt = self._yatt
            art.update()
            self._redraw()
