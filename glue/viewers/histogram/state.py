from __future__ import absolute_import, division, print_function

import numpy as np

from glue.core import Data

from glue.viewers.common.mpl_state import (MatplotlibDataViewerState,
                                           MatplotlibLayerState,
                                           DeferredDrawCallbackProperty)
from glue.core.state_objects import (StateAttributeLimitsHelper,
                                     StateAttributeHistogramHelper)

__all__ = ['HistogramViewerState', 'HistogramLayerState']


class HistogramViewerState(MatplotlibDataViewerState):

    xatt = DeferredDrawCallbackProperty()

    cumulative = DeferredDrawCallbackProperty(False)
    normalize = DeferredDrawCallbackProperty(False)

    hist_x_min = DeferredDrawCallbackProperty()
    hist_x_max = DeferredDrawCallbackProperty()
    hist_n_bin = DeferredDrawCallbackProperty(10)

    def __init__(self, **kwargs):
        super(HistogramViewerState, self).__init__(**kwargs)
        self.x_att_helper = StateAttributeLimitsHelper(self, 'xatt', lower='x_min',
                                                       upper='x_max', log='log_x')
        self.hist_helper = StateAttributeHistogramHelper(self, 'xatt', lower='hist_x_min',
                                                         upper='hist_x_max', n_bin='hist_n_bin')

    def update_priority(self, name):
        if name == 'layers':
            return 2
        elif name.endswith(('_min', '_max', '_bin')):
            return 0
        else:
            return 1

    def flip_x(self):
        self.x_att_helper.flip_limits()

    def _get_x_components(self):
        # Construct list of components over all layers
        components = []
        for layer_state in self.layers:
            if isinstance(layer_state.layer, Data):
                components.append(layer_state.layer.get_component(self.xatt))
            else:
                components.append(layer_state.layer.data.get_component(self.xatt))
        return components

    @property
    def bins(self):
        if self.log_x:
            return np.logspace(np.log10(self.hist_x_min),
                               np.log10(self.hist_x_max),
                               self.hist_n_bin + 1)
        else:
            return np.linspace(self.hist_x_min, self.hist_x_max,
                               self.hist_n_bin + 1)


class HistogramLayerState(MatplotlibLayerState):
    pass
