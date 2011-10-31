from cloudviz import Client
import matplotlib.pyplot as plt

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

    VizClient provides a public refresh() method that calls all of
    these methods. It also implements the _add_subset, _update_subset,
    _remove_subset, and _update_all methods from the Client class, by calling
    relevant methods above.

    Attributes:
    ----------

    _plots: A dictionary keyed by subset and data objects. Each entry
    holds the plot object associated with the key

    options: A dictionary of global plot options, to be handled by
             subclasses.

    """

    def __init__(self, data, options=None):
        Client.__init__(self, data)

        self._plots = {}
        if not options:
            self.options = {}
        else:
            self.options = options

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
        self._update_layer(s)
        self._redraw()

    def _update_subset(self, message):
        """
        Method to handle messages sent when subsets are modified.
        The plot properties of the modified subset are refreshed.

        """
        s = message.subset
        self._update_layer(s)
        self._redraw()

    def _remove_subset(self, message):
        """
        Method to handle messages sent when subsets are removed.

        """
        s = message.subset
        if s not in self._plots:
            return

        #remove from dictionary
        self._plots.pop(s)

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

    def _update_subset_plots(self):
        """
        Sync the location and visual properties
        of each point in each subset
        """
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
        raise NotImplementedError("VizClient Cannot Draw!")


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
            
