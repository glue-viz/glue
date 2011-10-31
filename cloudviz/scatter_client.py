import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

from cloudviz.viz_client import VizClient
from cloudviz.util import relim
import cloudviz as cv

class ScatterClient(VizClient):
    """
    A client class that uses matplotlib to visualize tables as scatter plots.

    The subset style dictionaries are passed directly to the
    scatter plot's 'set' method.
    """
    def __init__(self, data, figure=None, axes=None):
        VizClient.__init__(self, data)

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
        self.init_plot()
        
    def init_plot(self):
        for d in self.get_data():
            xatt = d.components.keys()[0]
            yatt = d.components.keys()[1]            
            self.init_layer(d, xatt=xatt, yatt=yatt)
            self.set_xdata(xatt, data=d)
            self.set_ydata(yatt, data=d)
            for s in d.subsets:
                self.init_layer(s, xatt=xatt, yatt=yatt)

    def init_layer(self, layer, xatt=None, yatt=None):

        # remove existing artist
        if layer in self.layers:
            artist = self.layers[layer]['artist']
            if artist in self.ax.collections:
                artist.remove()

        isSubset = isinstance(layer, cv.Subset)
        isData = isinstance(layer, cv.Data)

        if isData:
            data = layer
        else:
            data = layer.data

        if xatt is None:
            if isSubset:
                xatt = self.layers[data]['x']
            else:
                data.components.keys()[0]
        if yatt is None:
            if isSubset:
                yatt = self.layers[data]['y']
            else:
                yatt = data.components.keys()[1]

        x = data[xatt]
        y = data[yatt]

        if isinstance(layer, cv.Subset):
            ind = layer.to_index_list()
            if len(ind) == 0:
                x = [1e100]
                y = [1e100]
            else:
                x = x.flat[ind]
                y = y.flat[ind]
            
        artist = self.ax.scatter(x, y)
        artist.set_edgecolor('none')
        self.layers[layer] = {'artist':artist, 'x': xatt, 'y':yatt}

        if isSubset:
            artist.set_facecolor(layer.style.color)
            artist.set_alpha(layer.style.alpha)
    
    def update_layer(self, layer):
        print 'updating layer'
        if isinstance(layer, cv.Data):
            data = layer
            xatt = self.layers[layer]['x']
            yatt = self.layers[layer]['y']
            x = data[xatt]
            y = data[yatt]
        else: #layer is a subset
            data = layer.data
            ind = layer.to_index_list()
            xatt = self.layers[layer]['x']
            yatt = self.layers[layer]['y']
            x = data[xatt]
            y = data[yatt]
            x = x.flat[ind]
            y = y.flat[ind]
            artist = self.layers[layer]['artist']
            artist.set_facecolor(layer.style.color)
            artist.set_alpha(layer.style.alpha)

        xy = np.zeros((x.size, 2))
        xy[:, 0] = x
        xy[:, 1] = y
        self.layers[layer]['artist'].set_offsets(xy)
        self._redraw()

    def add_data(self, data):
        super(ScatterClient, self).add_data(data)
        self.init_layer(data)

    def _snap_xlim(self, data=None):
        """
        Reset the plotted x range to show all the data
        """
        if data is not None: 
            box = self.layers[data]['artist'].get_datalim(self.ax.transData)
        else:
            box = Bbox.union([l['artist'].get_datalim(self.ax.transData) 
                              for l in self.layers if isinstance(l, cv.Data)])
        range = relim(box.extents[0], box.extents[2], self.ax.get_xscale() == 'log')
        if self.ax.xaxis_inverted():
            range = [rage[1], range[0]]
            
        self.ax.set_xlim(range)

    def _snap_ylim(self, data=None):
        """
        Reset the plotted y range to show all the data
        """
        if data is not None: 
            box = self.layers[data]['artist'].get_datalim(self.ax.transData)
        else:
            box = Bbox.union([l['artist'].get_datalim(self.ax.transData) 
                              for l in self.layers if isinstance(l, cv.Data)])
        range = relim(box.extents[1], box.extents[3], self.ax.get_yscale() == 'log')
        if self.ax.yaxis_inverted():
            range = [range[1], range[0]]
        self.ax.set_ylim(range)

        
    def set_visible(self, layer, state):
        self.layers[layer].set_visible(state)

    def show(self, layer):
        self.layers[layer].set_visible(True)

    def hide(self, layer):
        self.layers[layer].set_visible(False)

        
    def set_xydata(self, coord, attribute, data=None):
        if data is None:
            data = self.data

        if coord not in ('x', 'y'):
            raise TypeError("coord must be one of x,y")

        if attribute not in data.components:
            raise KeyError("Invalid attribute: %s" % attribute)

        #update coordinates of data and subsets
        self.layers[data][coord] = attribute
        for s in data.subsets:
            if s not in self.layers: continue
            self.layers[s][coord] = attribute
            if coord == 'x':
                s.xatt = attribute
            else:
                s.yatt = attribute

        #update plots
        self.update_layer(data)
        map(self.update_layer, (l for l in self.layers if l in data.subsets))
    
        if coord == 'x':
            self._snap_xlim(data)
        else:
            self._snap_ylim(data)

        self.refresh()


    def set_xdata(self, attribute, data=None):
        """
        Redefine which component gets plotted on the x axis

        Parameters
        ----------
        attribute : string
                 The name of the new data component to plot
        data : class:`cloudviz.data.Data` instance (optional)
               which data set to apply to. Defaults to the first data set
               if not provided.
        """
        self.set_xydata('x', attribute, data=data)

    def set_ydata(self, attribute, data=None):
        """
        Redefine which component gets plotted on the y axis

        Parameters
        ----------
        attribute: string
                  The name of the new data component to plot

        data : class:`cloudviz.data.Data` instance (optional)
               which data set to apply to. Defaults to the first data set
               if not provided.
        """
        self.set_xydata('y', attribute, data=data)

    def _update_data_plot(self):
        self.update_layer(self.data)

    def _update_subset_single(self, s):
        self.update_layer(s)

    def _redraw(self):
        self.ax.figure.canvas.draw()

    def _update_axis_labels(self):
        pass
    
    def _add_subset(self, message):
        subset = message.sender
        subset.do_broadcast(False)
        self.init_layer(message.sender)
        subset.xatt = self.layers[subset]['x']
        subset.yatt = self.layers[subset]['y']
        subset.do_broadcast(True)

    def _update_subset(self, message):
        self.update_layer(message.sender)
