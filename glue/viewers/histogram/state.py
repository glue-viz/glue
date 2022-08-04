import numpy as np

from glue.core import BaseData, Subset

from echo import delay_callback
from glue.viewers.matplotlib.state import (MatplotlibDataViewerState,
                                           MatplotlibLayerState,
                                           DeferredDrawCallbackProperty as DDCProperty,
                                           DeferredDrawSelectionCallbackProperty as DDSCProperty)
from glue.core.state_objects import (StateAttributeLimitsHelper,
                                     StateAttributeHistogramHelper)
from glue.core.exceptions import IncompatibleAttribute, IncompatibleDataException
from glue.core.data_combo_helper import ComponentIDComboHelper
from glue.utils import defer_draw, datetime64_to_mpl
from glue.utils.decorators import avoid_circular

__all__ = ['HistogramViewerState', 'HistogramLayerState']


class HistogramViewerState(MatplotlibDataViewerState):
    """
    A state class that includes all the attributes for a histogram viewer.
    """

    x_att = DDSCProperty(docstring='The attribute to compute the histograms for')

    cumulative = DDCProperty(False, docstring='Whether to show the histogram as '
                                              'a cumulative histogram')
    normalize = DDCProperty(False, docstring='Whether to normalize the histogram '
                                             '(based on the total sum)')

    hist_x_min = DDCProperty(docstring='The minimum value used to compute the '
                                       'histogram')
    hist_x_max = DDCProperty(docstring='The maximum value used to compute the '
                                       'histogram')
    hist_n_bin = DDCProperty(docstring='The number of bins in the histogram')

    common_n_bin = DDCProperty(True, docstring='The number of bins to use for '
                                               'all numerical components')

    def __init__(self, **kwargs):

        super(HistogramViewerState, self).__init__()

        self.hist_helper = StateAttributeHistogramHelper(self, 'x_att', lower='hist_x_min',
                                                         upper='hist_x_max', n_bin='hist_n_bin',
                                                         common_n_bin='common_n_bin')

        self.x_lim_helper = StateAttributeLimitsHelper(self, 'x_att', lower='x_min',
                                                       upper='x_max', log='x_log')

        self.add_callback('layers', self._layers_changed)

        self.x_att_helper = ComponentIDComboHelper(self, 'x_att',
                                                   pixel_coord=True, world_coord=True)

        self.update_from_dict(kwargs)

        # This should be added after update_from_dict since we don't want to
        # influence the restoring of sessions.
        self.add_callback('hist_x_min', self.update_view_to_bins)
        self.add_callback('hist_x_max', self.update_view_to_bins)

        self.add_callback('x_log', self._reset_x_limits, priority=1000)

    def _reset_x_limits(self, *args):
        if self.x_att is None:
            return
        with delay_callback(self, 'hist_x_min', 'hist_x_max', 'x_min', 'x_max', 'x_log'):
            self.x_lim_helper.percentile = 100
            self.x_lim_helper.update_values(force=True)
            self.update_bins_to_view()

    def reset_limits(self):
        self._reset_x_limits()
        y_min = min(getattr(layer, '_y_min', np.inf) for layer in self.layers)
        if np.isfinite(y_min):
            self.y_min = y_min
        y_max = max(getattr(layer, '_y_max', 0) for layer in self.layers)
        if np.isfinite(y_max):
            self.y_max = y_max

    def _update_priority(self, name):
        if name == 'layers':
            return 2
        elif name.endswith('_log'):
            return 0.5
        elif name.endswith(('_min', '_max', '_bin')):
            return 0
        else:
            return 1

    def flip_x(self):
        """
        Flip the x_min/x_max limits.
        """
        self.x_lim_helper.flip_limits()

    @avoid_circular
    def update_bins_to_view(self, *args):
        """
        Update the bins to match the current view.
        """
        with delay_callback(self, 'hist_x_min', 'hist_x_max'):
            if self.x_max > self.x_min:
                self.hist_x_min = self.x_min
                self.hist_x_max = self.x_max
            else:
                self.hist_x_min = self.x_max
                self.hist_x_max = self.x_min

    @avoid_circular
    def update_view_to_bins(self, *args):
        """
        Update the view to match the histogram interval
        """
        with delay_callback(self, 'x_min', 'x_max'):
            self.x_min = self.hist_x_min
            self.x_max = self.hist_x_max

    @property
    def x_categories(self):
        return self._categories(self.x_att)

    def _categories(self, cid):

        categories = []

        for layer_state in self.layers:

            if isinstance(layer_state.layer, BaseData):
                layer = layer_state.layer
            else:
                layer = layer_state.layer.data

            try:
                if layer.data.get_kind(cid) == 'categorical':
                    categories.append(layer.data.get_data(cid).categories)
            except IncompatibleAttribute:
                pass

        if len(categories) == 0:
            return None
        else:
            return np.unique(np.hstack(categories))

    @property
    def x_kinds(self):
        return self._component_kinds(self.x_att)

    def _component_kinds(self, cid):

        # Construct list of component kinds over all layers

        kinds = set()

        for layer_state in self.layers:

            if isinstance(layer_state.layer, BaseData):
                layer = layer_state.layer
            else:
                layer = layer_state.layer.data

            try:
                kinds.add(layer.data.get_kind(cid))
            except IncompatibleAttribute:
                pass

        return kinds

    @property
    def bins(self):
        """
        The position of the bins for the histogram based on the current state.
        """

        if self.hist_x_min is None or self.hist_x_max is None or self.hist_n_bin is None:
            return None

        if self.x_log:
            return np.logspace(np.log10(self.hist_x_min),
                               np.log10(self.hist_x_max),
                               self.hist_n_bin + 1)
        elif isinstance(self.hist_x_min, np.datetime64):
            x_min = self.hist_x_min.astype(int)
            x_max = self.hist_x_max.astype(self.hist_x_min.dtype).astype(int)
            return np.linspace(x_min, x_max, self.hist_n_bin + 1).astype(self.hist_x_min.dtype)
        else:
            return np.linspace(self.hist_x_min, self.hist_x_max,
                               self.hist_n_bin + 1)

    @defer_draw
    def _layers_changed(self, *args):
        self.x_att_helper.set_multiple_data(self.layers_data)


class HistogramLayerState(MatplotlibLayerState):
    """
    A state class that includes all the attributes for layers in a histogram plot.
    """

    _histogram_cache = None

    def reset_cache(self, *args):
        self._histogram_cache = None

    @property
    def viewer_state(self):
        return self._viewer_state

    @viewer_state.setter
    def viewer_state(self, viewer_state):
        self._viewer_state = viewer_state

    @property
    def histogram(self):
        self.update_histogram()
        edges, unscaled = self._histogram_cache[1]
        scaled = unscaled.astype(float)
        dx = edges[1] - edges[0]
        if self.viewer_state.cumulative:
            scaled = scaled.cumsum()
            if self.viewer_state.normalize:
                scaled /= scaled.max()
        elif self.viewer_state.normalize:
            scaled /= (scaled.sum() * dx)
        return edges, scaled

    def update_histogram(self):

        current_settings = (id(self.viewer_state.x_att),
                            self.viewer_state.x_log,
                            self.viewer_state.hist_x_min,
                            self.viewer_state.hist_x_max,
                            self.viewer_state.hist_n_bin)

        if self._histogram_cache is not None and self._histogram_cache[0] == current_settings:
            return self._histogram_cache[1]

        if (self.viewer_state is None or self.viewer_state.x_att is None or
            self.viewer_state.hist_x_min is None or self.viewer_state.hist_x_max is None or
                self.viewer_state.hist_n_bin is None or self.viewer_state.x_log is None):
            raise IncompatibleDataException()

        if isinstance(self.layer, Subset):
            data = self.layer.data
            subset_state = self.layer.subset_state
        else:
            data = self.layer
            subset_state = None

        range = sorted((self.viewer_state.hist_x_min, self.viewer_state.hist_x_max))

        hist_values = data.compute_histogram([self._viewer_state.x_att],
                                             range=[range],
                                             bins=[self._viewer_state.hist_n_bin],
                                             log=[self._viewer_state.x_log],
                                             subset_state=subset_state)

        # TODO: determine whether this belongs here or in the layer artist
        if isinstance(range[0], np.datetime64):
            range = [datetime64_to_mpl(range[0]), datetime64_to_mpl(range[1])]

        if self._viewer_state.x_log:
            hist_edges = np.logspace(np.log10(range[0]), np.log10(range[1]),
                                     self._viewer_state.hist_n_bin + 1)
        else:
            hist_edges = np.linspace(range[0], range[1],
                                     self._viewer_state.hist_n_bin + 1)

        self._histogram_cache = current_settings, (hist_edges, hist_values)
