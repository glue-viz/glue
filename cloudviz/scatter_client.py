from cloudviz.viz_client import VizClient


class ScatterClient(VizClient):
    """
    A client class that uses matplotlib to visualize tables as scatter plots.

    The subset style dictionaries are passed directly to the
    scatter plot's 'set' method.
    """
    def __init__(self, data):
        VizClient.__init__(self, data)

        # name of the attribute on the x axis
        self._xatt = None

        # name of the attribute on the y axis
        self._yatt = None

        # numerical data on the x axis
        self._xdata = None

        # numerical data on the y axis
        self._ydata = None

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
