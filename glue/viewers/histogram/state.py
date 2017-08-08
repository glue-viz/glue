from __future__ import absolute_import, division, print_function

import numpy as np

from glue.core import Data

from glue.external.echo import delay_callback
from glue.viewers.matplotlib.state import (MatplotlibDataViewerState,
                                           MatplotlibLayerState,
                                           DeferredDrawCallbackProperty as DDCProperty,
                                           DeferredDrawSelectionCallbackProperty as DDSCProperty)
from glue.core.state_objects import (StateAttributeLimitsHelper,
                                     StateAttributeHistogramHelper)
from glue.core.exceptions import IncompatibleAttribute
from glue.core.data_combo_helper import ComponentIDComboHelper
from glue.utils import defer_draw

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
    hist_x_max = DDCProperty(docstring='The maxumum value used to compute the '
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

        self.x_att_helper = ComponentIDComboHelper(self, 'x_att')

        self.update_from_dict(kwargs)

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

    def update_bins_to_view(self):
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

    def _get_x_components(self):

        if self.x_att is None:
            return []

        # Construct list of components over all layers

        components = []

        for layer_state in self.layers:

            if isinstance(layer_state.layer, Data):
                layer = layer_state.layer
            else:
                layer = layer_state.layer.data

            try:
                components.append(layer.get_component(self.x_att))
            except IncompatibleAttribute:
                pass

        return components

    @property
    def bins(self):
        """
        The position of the bins for the histogram based on the current state.
        """
        if self.x_log:
            return np.logspace(np.log10(self.hist_x_min),
                               np.log10(self.hist_x_max),
                               self.hist_n_bin + 1)
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
