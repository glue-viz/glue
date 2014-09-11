from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
from ..core.client import Client
from ..core import Data
from .layer_artist import LayerArtistContainer


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


def init_mpl(figure, axes, wcs=False, axes_factory=None):
    if axes is not None and figure is not None and \
            axes.figure is not figure:
        raise ValueError("Axes and figure are incompatible")

    try:
        from ..external.wcsaxes import WCSAxesSubplot
    except ImportError:
        WCSAxesSubplot = None

    if axes is not None:
        _ax = axes
        _figure = axes.figure
    else:
        _figure = figure or plt.figure()
        if wcs and WCSAxesSubplot is not None:
            _ax = WCSAxesSubplot(_figure, 111)
            _figure.add_axes(_ax)
        else:
            if axes_factory is not None:
                _ax = axes_factory(_figure)
            else:
                _ax = _figure.add_subplot(1, 1, 1)
    try:
        _figure.set_tight_layout(True)
    except AttributeError:  # matplotlib < 1.1
        pass

    return _figure, _ax


class GenericMplClient(Client):

    """
    This client base class handles the logic of adding, removing,
    and updating layers.

    Subsets are auto-added and removed with datasets.
    New subsets are auto-added iff the data has already been added
    """

    def __init__(self, data=None, figure=None, axes=None,
                 artist_container=None):

        super(GenericMplClient, self).__init__(data=data)
        figure, self.axes = init_mpl(figure, axes,
                                     axes_factory=self.create_axes)
        self.artists = artist_container
        if self.artists is None:
            self.artists = LayerArtistContainer()

        self._connect()

    def create_axes(self, figure):
        return figure.add_subplot(1, 1, 1)

    def _connect(self):
        pass

    @property
    def collect(self):
        # a better name
        return self.data

    def _redraw(self):
        self.axes.figure.canvas.draw()

    def new_layer_artist(self, layer):
        raise NotImplementedError

    def apply_roi(self, roi):
        raise NotImplementedError

    def _update_layer(self, layer):
        raise NotImplementedError

    def add_layer(self, layer):
        """
        Add a new Data or Subset layer to the plot.

        Returns the created layer artist

        :param layer: The layer to add
        :type layer: :class:`~glue.core.Data` or :class:`~glue.core.Subset`
        """
        if layer.data not in self.collect:
            return

        if layer in self.artists:
            return self.artists[layer][0]

        result = self.new_layer_artist(layer)
        self.artists.append(result)
        self._update_layer(layer)

        self.add_layer(layer.data)
        for s in layer.data.subsets:
            self.add_layer(s)

        if layer.data is layer:  # Added Data object. Relimit view
            self.axes.autoscale_view(True, True, True)

        return result

    def remove_layer(self, layer):
        if layer not in self.artists:
            return

        self.artists.pop(layer)
        if isinstance(layer, Data):
            map(self.remove_layer, layer.subsets)

        self._redraw()

    def set_visible(self, layer, state):
        """
        Toggle a layer's visibility

        :param layer: which layer to modify
        :param state: True or False
        """

    def _update_all(self):
        for layer in self.artists.layers:
            self._update_layer(layer)

    def __contains__(self, layer):
        return layer in self.artists

    # Hub message handling
    def _add_subset(self, message):
        self.add_layer(message.sender)

    def _remove_subset(self, message):
        self.remove_layer(message.sender)

    def _update_subset(self, message):
        self._update_layer(message.sender)

    def _update_data(self, message):
        self._update_layer(message.sender)

    def _remove_data(self, message):
        self.remove_layer(message.data)

    def restore_layers(self, layers, context):
        """ Re-generate plot layers from a glue-serialized list"""
        for l in layers:
            l.pop('_type')
            props = dict((k, context.object(v)) for k, v in l.items())
            layer = self.add_layer(props['layer'])
            layer.properties = props
