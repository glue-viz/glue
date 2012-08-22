import matplotlib.pyplot as plt
from ..core.client import Client


class VizClient(Client):
    """
    The VizClient class provides an interface (and minimal
    implementation) for a generic client that creates
    visualizations. The goal of VizClient is to provide a reusable way
    to organize client plotting code.

    Clients which extend VizClient should override the following methods
    to perform specific visualization tasks

    * _update_axis_labels
    * _update_data_plot
    * _update_subset_single
    * _redraw
    * init_layer

    VizClient provides a public refresh() method that calls all of
    these methods.

    Attributes:
    ----------

    options: A dictionary of global plot options, to be handled by
             subclasses.

    """

    def __init__(self, data, options=None):
        Client.__init__(self, data)

        if not options:
            self.options = {}
        else:
            self.options = options

    def _add_data(self, message):
        pass

    def _remove_data(self, message):
        pass

    def _update_data(self, message):
        """
        Method to handle messages sent by the dataset. Refreshes the display.
        """
        self._update_data_plot()
        self.refresh()

    def _add_subset(self, message):
        """
        Method to handle messages sent when subsets are created.
        """
        s = message.subset
        self.init_layer(s)
        self._redraw()

    def _update_subset(self, message):
        """
        Method to handle messages sent when subsets are modified.
        The plot properties of the modified subset are refreshed.

        """
        s = message.subset
        self._update_subset_single(s, redraw=True)

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
        raise NotImplementedError("VizClient cannot draw!")

    def _update_axis_labels(self):
        """
        Sync the axis labels to reflect which components are
        currently being plotted
        """
        raise NotImplementedError("VizClient cannot draw!")

    def _update_data_plot(self):
        """
        Sync the location of the scatter points to
        reflect what components are being plotted
        """
        raise NotImplementedError("VizClient cannot draw!")

    def _update_subset_plots(self, redraw=False):
        """
        Sync the location and visual properties
        of each point in each subset
        """
        junk = [self._update_subset_single(s) for d in self.data
                for s in d.subsets]
        if redraw:
            self._redraw()

    def _update_subset_single(self, s, redraw=False):
        """
        Update the properties of a subset

        Parameters
        ----------
        s: A subset instance
        The subset to refresh.

        """
        raise NotImplementedError("VizClient Cannot Draw!")

    def init_layer(self, layer):
        """Initialize a plot of a data or subset object for the first time.

        Parameters
        ----------
        layer: Data or subset instance
        """
        raise NotImplementedError()


def init_mpl(figure, axes):
    if axes is not None and figure is not None and \
            axes.figure is not figure:
        raise Exception("Axes and figure are incompatible")

    if axes is not None:
        _ax = axes
        _figure = axes.figure
    else:
        if figure is None:
            _figure = plt.figure()
        _ax = _figure.add_subplot(1, 1, 1)

    return _figure, _ax
