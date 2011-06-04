from cloudviz import Client


class ScatterClient(Client):
    """
    A client class that uses matplotlib to visualize tables as scatter plots.

    The subset style dictionaries are passed directly to the
    scatter plot's 'set' method.
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

    def _update_all(self, message):
        """
        Method to handle messages sent by the dataset. Refreshes the display.
        """
        self.refresh()

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
        raise NotImplementedError("base CatalogClient cannot draw!")

    def _update_axis_labels(self):
        """
        Sync the axis labels to reflect which components are
        currently being plotted
        """
        raise NotImplementedError("Base CatalogClient cannot draw!")

    def _update_data_plot(self):
        """
        Sync the location of the scatter points to
        reflect what components are being plotted
        """
        raise NotImplementedError("Base CatalogClient cannot draw!")

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
        raise NotImplementedError("Base Catalog Class Cannot Draw!")

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
            raise AttributeError("attribute must be a valid "
                                 "component in the data")

        if axis == 'x':
            if self._xatt == attribute:
                return
            self._xatt = attribute
            self._xdata = self.data.components[attribute].data.ravel()
        if axis == 'y':
            if self._yatt == attribute:
                return
            self._yatt = attribute
            self._ydata = self.data.components[attribute].data.ravel()


