from __future__ import absolute_import, division, print_function

import numpy as np

from glue.core import Data

from glue.viewers.matplotlib.state import (MatplotlibDataViewerState,
                                           MatplotlibLayerState,
                                           DeferredDrawCallbackProperty)
from glue.core.state_objects import (StateAttributeLimitsHelper,
                                     StateAttributeHistogramHelper)

__all__ = ['HistogramViewerState', 'HistogramLayerState']


class HistogramViewerState(MatplotlibDataViewerState):

    x_att = DeferredDrawCallbackProperty()

    cumulative = DeferredDrawCallbackProperty(False)
    normalize = DeferredDrawCallbackProperty(False)

    hist_x_min = DeferredDrawCallbackProperty()
    hist_x_max = DeferredDrawCallbackProperty()
    hist_n_bin = DeferredDrawCallbackProperty()

    common_n_bin = DeferredDrawCallbackProperty(True)

    def __init__(self, **kwargs):
        super(HistogramViewerState, self).__init__(**kwargs)
        self.x_att_helper = StateAttributeLimitsHelper(self, 'x_att', lower='x_min',
                                                       upper='x_max', log='x_log')
        self.hist_helper = StateAttributeHistogramHelper(self, 'x_att', lower='hist_x_min',
                                                         upper='hist_x_max', n_bin='hist_n_bin',
                                                         common_n_bin='common_n_bin')

    def update_priority(self, name):
        if name == 'layers':
            return 2
        elif name.endswith(('_min', '_max', '_bin')):
            return 0
        else:
            return 1

    def flip_x(self):
        self.x_att_helper.flip_limits()

    def update_bins_to_view(self):
        # TODO: delay callback
        if self.x_max > self.x_min:
            self.hist_x_min = self.x_min
            self.hist_x_max = self.x_max
        else:
            self.hist_x_min = self.x_max
            self.hist_x_max = self.x_min


    def _get_x_components(self):
        # Construct list of components over all layers
        components = []
        for layer_state in self.layers:
            if isinstance(layer_state.layer, Data):
                components.append(layer_state.layer.get_component(self.x_att))
            else:
                components.append(layer_state.layer.data.get_component(self.x_att))
        return components

    @property
    def bins(self):
        if self.x_log:
            return np.logspace(np.log10(self.hist_x_min),
                               np.log10(self.hist_x_max),
                               self.hist_n_bin + 1)
        else:
            return np.linspace(self.hist_x_min, self.hist_x_max,
                               self.hist_n_bin + 1)


class HistogramLayerState(MatplotlibLayerState):
    pass
