import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

from cloudviz.viz_client import VizClient
from cloudviz.util import relim
import cloudviz as cv


class ScatterClient(VizClient):
    """
    A client class that uses matplotlib to visualize tables as scatter plots.
    """
    def __init__(self, data=None, figure=None, axes=None):
        """
        Create a new ScatterClient object

        Inputs:
        =======
        data : `cloudviz.data.Data` instance, DataCollection, or a list of data
             Initial data to show
        figure : matplotlib Figure instance (optional)
           Which figure instance to draw to. One will be created if not provided
        axes : matplotlib Axes instance (optional)
           Which axes instance to use. Will be created if necessary
        """
        VizClient.__init__(self, data=data)

        #layers keyed by layer (data/subset) objects.
        # values are dicts, with keys:
        #   'artist': matplotlib artist
        #   'x' : xattribute
        #   'y' : yattribute
        #   'attributes' : list of plottable attributes (components of data)
        self.layers = {}

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

        #each data set and subset is stored is a layer
        for d in self.data:
            self.init_layer(d)
            for s in d.subsets:
                self.init_layer(s)

    def init_layer(self, layer, xatt=None, yatt=None):
        """ Adds a new visual layer to a client, to display either a dataset
        or a subset. Updates both the client data structure and the
        plot.

        Inputs:
        =======
        layer : `cloudviz.data.Data` or `cloudviz.subset.Subset` object
            The layer to add

        xatt : string (optional)
            The attribute to map onto the x axis
        yatt : string (optional)
            The attribute to map onto the y axis
        """

        # remove existing artist
        if layer in self.layers:
            artist = self.layers[layer]['artist']
            if artist in self.ax.collections:
                artist.remove()

        data = layer.data
        attributes = [c for c in data.components if
                      np.can_cast(data[c].dtype, np.float)]

        if xatt is None:
            xatt = attributes[0]
            if data in self.layers:
                xatt = self.layers[data]['x']
        if yatt is None:
            yatt = attributes[1]
            if data in self.layers:
                yatt = self.layers[data]['y']

        x = data[xatt]
        y = data[yatt]

        empty = False
        if isinstance(layer, cv.Subset):
            ind = layer.to_index_list()
            x = x.flat[ind]
            y = y.flat[ind]
            if x.size == 0:
                x = [1]
                y = [1]
                empty = True

        artist = self.ax.scatter(x, y)
        if empty:
            artist.set_offsets(np.zeros((0, 2)))

        artist.set_edgecolor('none')

        self.layers[layer] = {'artist': artist,
                              'x': xatt, 'y': yatt,
                              'attributes': attributes}

        self._sync_visual(artist, layer.style)

    def _sync_visual(self, plot, style):
        """ Make sure that each dataset / subset's style
        property is accurately reflected in the visualization """
        plot.set_facecolor(style.color)
        try:
            plot.get_sizes().data[0] = style.markersize
        except TypeError:
            plot.get_sizes()[0] = style.markersize

        plot.set_alpha(style.alpha)

    def update_layer(self, layer):
        """ Update the matplotlib artist for the requested layer """
        if isinstance(layer, cv.Data):
            data = layer
            xatt = self.layers[layer]['x']
            yatt = self.layers[layer]['y']
            x = data[xatt]
            y = data[yatt]
        else:  # Layer is a subset
            data = layer.data
            ind = layer.to_index_list()
            xatt = self.layers[layer]['x']
            yatt = self.layers[layer]['y']
            x = data[xatt]
            y = data[yatt]
            x = x.flat[ind]
            y = y.flat[ind]

        xy = np.zeros((x.size, 2))
        xy[:, 0] = x
        xy[:, 1] = y
        artist = self.layers[layer]['artist']
        artist.set_offsets(xy)
        self._sync_visual(artist, layer.style)
        self._redraw()

    def add_data(self, data):
        """ Add a new data set. Called automatically by hub """
        super(ScatterClient, self).add_data(data)
        self.init_layer(data)

    def _snap_xlim(self, data=None):
        """
        Reset the plotted x range to show all the data

        Inputs:
        =======
        data : `cloudviz.data.Data` instance
             If provided, will snap using values in this data set
        """
        if data is None:
            data = self.data[0]
        xy = self.layers[data]['artist'].get_offsets()
        range = relim(min(xy[:, 0]), max(xy[:, 0]), self.ax.get_xscale() == 'log')
        if self.ax.xaxis_inverted():
            range = [rage[1], range[0]]

        self.ax.set_xlim(range)

    def _snap_ylim(self, data=None):
        """
        Reset the plotted y range to show all the data

        Inputs:
        =======
        data : `cloudviz.data.Data` instance
             If provided, will snap using values in this data set
        """
        if data is None:
            data = self.data
        xy = self.layers[data]['artist'].get_offsets()
        range = relim(min(xy[:, 1]), max(xy[:, 1]),
                      self.ax.get_yscale() == 'log')
        if self.ax.yaxis_inverted():
            range = [range[1], range[0]]
        self.ax.set_ylim(range)

    def set_visible(self, layer, state):
        """ Toggle a layer's visibility

        Inputs:
        =======
        layer : `cloudviz.data.Data` or `cloudviz.subset.Subset` instance
              Which layer to modify
        state : boolean
              True to show. False to hide
        """
        self.layers[layer]['artist'].set_visible(state)
        self._redraw()

    def get_attributes(self, layer):
        return self.layers[layer]['attributes']

    def current_x_attribute(self, layer):
        return self.layers[layer]['x']

    def current_y_attribute(self, layer):
        return self.layers[layer]['y']

    def show(self, layer):
        """ Show a layer
        Inputs:
        =======
        layer : `cloudviz.data.Data` or `cloudviz.subset.Subset` instance
              Which layer to modify
        """
        self.layers[layer]['artist'].set_visible(True)

    def hide(self, layer):
        """ Hide a layer
        Inputs:
        =======
        layer : `cloudviz.data.Data` or `cloudviz.subset.Subset` instance
              Which layer to modify
        """
        self.layers[layer].set_visible(False)

    def set_xydata(self, coord, attribute, data=None, snap=True):
        """ Redefine which components get assigned to the x/y axes

        Inputs:
        =======
        coord : 'x' or 'y'
           Which axis to reassign
        attribute : string
           Which attribute of the data to use.
        data : `cloudviz.data.Data` instance
           Which dataset to use. Defaults to first data set added to client
        snap : bool
           If True, will rescale x/y axes to fit the data
        """

        if data is None:
            data = self.data.data[0]

        if coord not in ('x', 'y'):
            raise TypeError("coord must be one of x,y")

        if attribute not in data.components:
            raise KeyError("Invalid attribute: %s" % attribute)

        #update coordinates of data and subsets
        self.layers[data][coord] = attribute
        for s in data.subsets:
            if s not in self.layers:
                continue
            self.layers[s][coord] = attribute
            if coord == 'x':
                s.xatt = attribute
            else:
                s.yatt = attribute

        #update plots
        self.update_layer(data)
        map(self.update_layer, (l for l in self.layers if l in data.subsets))

        if coord == 'x' and snap:
            self._snap_xlim(data)
        elif coord == 'y' and snap:
            self._snap_ylim(data)

        self.refresh()

    def set_xdata(self, attribute, data=None, snap=True):
        """
        Redefine which component gets plotted on the x axis

        Parameters
        ----------
        attribute : string
                 The name of the new data component to plot
        data : class:`cloudviz.data.Data` instance (optional)
               which data set to apply to. Defaults to the first data set
               if not provided.
        snap : bool
             If true, re-scale x axis to show all values
        """
        self.set_xydata('x', attribute, data=data, snap=snap)

    def set_ydata(self, attribute, data=None, snap=True):
        """
        Redefine which component gets plotted on the y axis

        Parameters
        ----------
        attribute: string
                  The name of the new data component to plot

        data : class:`cloudviz.data.Data` instance (optional)
               which data set to apply to. Defaults to the first data set
               if not provided.

        snap : bool
               If True, re-scale y axis to show all values
        """
        self.set_xydata('y', attribute, data=data, snap=snap)

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

    def set_xflip(self, state):
        """ Set whether the x axis increases or decreases to the right.

        Inputs:
        =======
        state : bool
            True to flip x axis
        """
        range = self.ax.get_xlim()
        if state:
            self.ax.set_xlim(max(range), min(range))
        else:
            self.ax.set_xlim(min(range), max(range))
        self._redraw()

    def set_yflip(self, state):
        range = self.ax.set_ylim()
        if state:
            self.ax.set_ylim(max(range), min(range))
        else:
            self.ax.set_ylim(min(range), max(range))
        self._redraw()

    def _remove_data(self, message):
        for s in message.data.subsets:
            self.delete_layer(s)
        self.delete_layer(message.data)

    def _remove_subset(self, message):
        self.delete_layer(message.subset)

    def delete_layer(self, layer):
        artist = self.layers[layer]['artist']
        print artist, artist in self.ax.collections
        if artist in self.ax.collections:
            artist.remove()
            self._redraw()

    def _update_data_plot(self):
        [self.update_layer(d) for d in self.data]

    def _update_subset_single(self, s):
        self.update_layer(s)

    def _redraw(self):
        self.ax.figure.canvas.draw()

    def _update_axis_labels(self):
        pass

    def _add_subset(self, message):
        subset = message.sender
        subset.do_broadcast(False)
        if not isinstance(subset, cv.subset.RoiSubset):
            raise TypeError("Only ROI subsets supported")

        self.init_layer(message.subset)
        self.active_layer = subset

        subset.xatt = self.layers[subset]['x']
        subset.yatt = self.layers[subset]['y']
        subset.do_broadcast(True)

    def _update_subset(self, message):
        self.update_layer(message.sender)

    def _add_data(self, data):
        super(ScatterClient, self)._add_data(data)
        self.init_layer(data)
        for s in data.subsets:
            self.init_layer(s)
