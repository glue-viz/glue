from cloudviz import Client
import matplotlib.pyplot as plt
import numpy as np

class CatalogClient(Client):
    """ 
    A client class that uses matplotlib to visualize tables as scatter plots.

    The subset style dictionaries are passed directly to the scatter plot's 'set'
    method.
    """
    def __init__(self, data):
        Client.__init__(self, data)
        
        # name of the attribute on the x axis
        self._xatt = None

        # name of the attribute on the y axis
        self._yatt = None

        # numerical data on the x axis
        self._xdata = None

        # numerical data on the y axis
        self._ydata = None

        # dictionary of all the scatter plots. 
        # Keyed by data/subset objects
        self._scatter = {}
        
        # figure object
        self._figure = plt.figure()

        # plot/axes object
        self._ax = self._figure.add_subplot(1,1,1)

    def _update_all(self, message):
        """ 
        Method to handle messages sent by the dataset. Refreshes the display.
        """
        refresh()

    def _add_subset(self, message):
        """ 
        Method to handle messages sent when subsets are created.
        """
        s = message.subset
        self._update_subset_single(s)
        self._redraw()

    def _update_subset(self, message):
        """ 
        Method to handle messages sent when subsets are modified.
        The plot properties of the modified subset are refreshed.

        """
        s = message.subset
        self._update_subset_single(s)
        self._redraw()

    def _remove_subset(self, message):
        """ 
        Method to handle messages sent when subsets are removed.

        """
        s = message.subset
        if s not in self._scatter:
            return

        #remove from plot
        self._scatter[s].remove()

        #remove from dictionary
        self._scatter.pop(s)
        
        self._redraw()

    def refresh(self):
        """ 
        Update and redraw all plot information.
        """
        self._update_data_plot()
        self._update_subset_plots()
        self._update_axis_labels()
        self._redraw()
    
    def _redraw(self):
        """ 
        Redraw, but do not update, plot information
        """
        self._figure.canvas.draw()

    def _update_axis_labels(self):
        """ 
        Sync the axis labels to reflect which components are
        currently being plotted
        """
        xlabel = self._xatt
        if xlabel is not None and \
                self.data.components[self._xatt].units is not None:
            xlabel += ' (%s)' % self.data.components[self._xatt].units

        ylabel = self._yatt
        if ylabel is not None and \
                self.data.components[self._yatt].units is not None:
            ylabel += ' (%s)' % self.data.components[self._yatt].units

        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)

    def _update_data_plot(self):
        """ 
        Sync the location of the scatter points to
        reflect what components are being plotted
        """
        if self._xdata is None or self._ydata is None: 
            return
        if self.data not in self._scatter:
            plot = self._ax.scatter(self._xdata, self._ydata, color='k')
            self._scatter[self.data] = plot
        else:
            self._scatter[self.data].set_offsets(
                zip(self._xdata, self._ydata))

    def _update_subset_plots(self):
        """
        Sync the location and visual properties
        of each point in each subset
        """
        if self._xdata is None or self._ydata is None:
            return

        for s in self.data.subsets:
            self._update_subset_single(s)

    def _update_subset_single(self, s):
        """
        Update the location and visual properties
        of each point in a single subset

        Parameters
        ----------
        s: A subset instance
        The subset to refresh.

        """
        if s not in self.data.subsets:
            raise Exception("Input is not one of data's subsets: %s" % s)

        if s not in self._scatter:
            plot = self._ax.scatter(self._xdata[s.mask],
                              self._ydata[s.mask], s=5)
            self._scatter[s] = plot
        else:
            self._scatter[s].set_offsets(
                zip(self._xdata[s.mask], self._ydata[s.mask]))
        self._scatter[s].set(**s.style)
                    
    def set_xdata(self, attribute):
        """ 
        Redefine which component gets plotted on the x axis

        Parameters
        ----------
        attribute: string
                 The name of the new data component to plot
        """
        self._set_attribute(attribute, axis='x')
        self.refresh()

    def set_ydata(self, attribute):
        """
        Redefine which component gets plotted on the y axis

        Parameters
        ----------
        attribute: string
                  The name of the new data component to plot
        """
        self._set_attribute(attribute, axis='y')
        self.refresh()

    def _set_attribute(self, attribute, axis=None):
        """
        Redefine which data are plotted on each axis
        
        Parameters
        ----------
        attribute: string
                 The name of the new data component to use
        axis: 'x' or 'y'
              Which axis to assign attribute to

        """

        if axis not in ['x', 'y']:
            raise AttributeError("axis must be one of 'x', 'y'")
        if attribute not in self.data.components:
            raise AttributeError("attribute must be a valid component in the data")

        if axis=='x':
            if self._xatt == attribute: 
                return
            self._xatt = attribute
            self._xdata = self.data.components[attribute].data.ravel()
        if axis=='y':
            if self._yatt == attribute:
                return
            self._yatt = attribute
            self._ydata = self.data.components[attribute].data.ravel()


if __name__ == "__main__":
    """
    This is a self contained (non-interactive) example of setting up
    a cloudviz environment with 2 catalog clients linked to the same data.
    """
    import cloudviz as cv
    from time import sleep
    import sys

    #set up the data
    d = cv.TabularData()
    d.read_data(sys.path[0]+'/oph_c2d_yso_catalog.tbl')

    # create the hub
    h = cv.Hub()

    # create the 2 clients
    c = CatalogClient(d)
    c2 = CatalogClient(d)

    # register the clients and data to the hub
    # (to receive and send events, respectively)
    c.register_to_hub(h)
    c2.register_to_hub(h)
    d.register_to_hub(h)

    #define the axes to plot
    c.set_xdata('ra')
    c.set_ydata('dec')
    c2.set_xdata('IR1_flux_1')
    c2.set_ydata('IR2_flux_1')
    sleep(1)

    # create a new subset. Note we need to register() each subset
    s = cv.subset.ElementSubset(d)
    mask = d.components['ra'].data.ravel() > 248
    s.mask = mask
    s.register()
    sleep(1)

    #change plot properties. This gets updated autmatically
    s.style['color'] = 'm'
    s.style['alpha'] = .8
    sleep(1)

    #change one of the axes
    c2.set_ydata('IR3_flux_1')
    sleep(1)
