import matplotlib.pyplot as plt
import numpy as np

from glue.client import Client
from glue.util import relim
from glue.exceptions import InvalidView

class ScatterClient(Client):
    """
    A client class that uses matplotlib to visualize tables as scatter plots.
    """
    def __init__(self, data=None, figure=None, axes=None):
        """
        Create a new ScatterClient object

        Inputs:
        =======
        data : `glue.data.Data` instance, DataCollection, or a list of data
        Initial data to show

        figure : matplotlib Figure instance (optional)
           Which figure instance to draw to. One will be created if
           not provided

        axes : matplotlib Axes instance (optional)
           Which axes instance to use. Will be created if necessary
        """
        Client.__init__(self, data=data)

        #layers keyed by layer (data/subset) objects.
        # values are dicts, with keys:
        #   'artist': matplotlib artist
        #   'attributes' : list of plottable attributes (components of data)
        self.layers = {}

        self.xatt = None
        self.yatt = None
        self._layer_updated = False

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

    def _set_layer_enabled(self, layer, enabled):
        """ Enable/disable a layer. Disabled layers are invisible,
        and do not updated when the plot is redrawn.

        Layers are enabled/disabled when encountering invalid
        plot requests (I.e. plotting an attribute that doesn't
        exist for a data set."""
        if layer not in self.layers:
            return
        self.set_visible(layer, enabled)
        self.layers[layer]['enabled'] = enabled


    def is_layer_present(self, layer):
        """ True if layer is plotted """
        return layer in self.layers and 'artist' in self.layers[layer]


    def plottable_attributes(self, data):
        return [c for c in data.components if
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

        self._remove_layer_artists(layer)

        data = layer.data
        attributes = self.plottable_attributes(data)

        if self.xatt is None:
            self.xatt = attributes[0]

        if self.yatt is None:
            self.yatt = attributes[1]

        artist = self.ax.scatter([1], [1])
        artist.set_offsets(np.zeros((0, 2)))

        self.layers[layer] = {'artist': artist,
                              'attributes': attributes}
        self._update_layer(layer)

    def _sync_style(self, layer):
        """ Make sure that each layer's style
        property is accurately reflected in the visualization """
        artist = self.layers[layer]['artist']
        style = layer.style

        artist.set_edgecolor('none')
        artist.set_facecolor(style.color)
        try:
            artist.get_sizes().data[0] = style.markersize
        except TypeError:
            artist.get_sizes()[0] = style.markersize

        artist.set_alpha(style.alpha)

    def _snap_xlim(self):
        """
        Reset the plotted x rng to show all the data
        """
        rng = [np.infty, -np.infty]
        is_log = self.ax.get_xscale() == 'log'
        for layer in self.layers:
            artist = self.layers[layer]['artist']
            if not artist.get_visible():
                continue
            xy = artist.get_offsets()
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
        for layer in self.layers:
            artist = self.layers[layer]['artist']
            if not artist.get_visible():
                continue
            xy = artist.get_offsets()
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
        if layer not in self.layers or 'artist' not in self.layers[layer]:
            return
        self.layers[layer]['artist'].set_visible(state)
        self._redraw()


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
            self.xatt = attribute
        elif coord == 'y':
            self.yatt = attribute

        #update plots
        map(self._update_layer, (l for l in self.layers))

        if coord == 'x' and snap:
            self._snap_xlim()
        elif coord == 'y' and snap:
            self._snap_ylim()

        self._update_axis_labels()

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
        if layer not in self.layers:
            return
        artist = self.layers[layer]['artist']
        if artist in self.ax.collections:
            artist.remove()

    def _remove_data(self, message):
        for s in message.data.subsets:
            self.delete_layer(s)
        self.delete_layer(message.data)

    def _remove_subset(self, message):
        self.delete_layer(message.subset)

    def delete_layer(self, layer):
        if layer not in self.layers:
            return
        artist = self.layers[layer]['artist']
        if artist in self.ax.collections:
            artist.remove()
        self.layers.pop(layer)
        self._redraw()

    def _update_data(self, message):
        data = message.sender
        self._update_layer(data)

    def _redraw(self):
        self.ax.figure.canvas.draw()

    def _update_axis_labels(self):
        self.ax.set_xlabel(self.xatt)
        self.ax.set_ylabel(self.yatt)

    def _add_subset(self, message):
        subset = message.sender
        subset.do_broadcast(False)
        self.add_layer(subset)
        subset.do_broadcast(True)

    def _update_subset(self, message):
        self._update_layer(message.sender)

    def _update_layer(self, layer):
        """ Update both the style and data for the requested layer"""
        if self.xatt is None or self.yatt is None:
            return

        if layer not in self.layers:
            return

        try:
            x = layer[self.xatt]
            y = layer[self.yatt]
        except InvalidView:
            self._set_layer_enabled(layer, False)
            return

        self._layer_updated = True

        self._set_layer_enabled(layer, True)
        xy = np.zeros((x.size, 2))
        xy[:, 0] = x
        xy[:, 1] = y
        artist = self.layers[layer]['artist']
        artist.set_offsets(xy)
        self._sync_style(layer)
        self._redraw()
