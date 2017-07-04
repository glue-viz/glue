from __future__ import absolute_import, division, print_function

from glue.core import Data

from glue.viewers.matplotlib.state import (MatplotlibDataViewerState,
                                           MatplotlibLayerState,
                                           DeferredDrawCallbackProperty as DDCProperty)
from glue.core.state_objects import StateAttributeLimitsHelper
from glue.external.echo import keep_in_sync

__all__ = ['ScatterViewerState', 'ScatterLayerState']


class ScatterViewerState(MatplotlibDataViewerState):
    """
    A state class that includes all the attributes for a scatter viewer.
    """

    x_att = DDCProperty(docstring='The attribute to show on the x-axis')
    y_att = DDCProperty(docstring='The attribute to show on the y-axis')

    def __init__(self, **kwargs):

        super(ScatterViewerState, self).__init__(**kwargs)

        self.limits_cache = {}

        self.x_att_helper = StateAttributeLimitsHelper(self, attribute='x_att',
                                                       lower='x_min', upper='x_max',
                                                       log='x_log',
                                                       limits_cache=self.limits_cache)

        self.y_att_helper = StateAttributeLimitsHelper(self, attribute='y_att',
                                                       lower='y_min', upper='y_max',
                                                       log='y_log',
                                                       limits_cache=self.limits_cache)

    def _update_priority(self, name):
        if name == 'layers':
            return 2
        elif name.endswith(('_min', '_max', '_log')):
            return 0
        else:
            return 1

    def flip_x(self):
        """
        Flip the x_min/x_max limits.
        """
        self.x_att_helper.flip_limits()

    def flip_y(self):
        """
        Flip the y_min/y_max limits.
        """
        self.y_att_helper.flip_limits()

    def _get_x_components(self):
        return self._get_components(self.x_att)

    def _get_y_components(self):
        return self._get_components(self.y_att)

    def _get_components(self, cid):
        # Construct list of components over all layers
        components = []
        for layer_state in self.layers:
            if isinstance(layer_state.layer, Data):
                components.append(layer_state.layer.get_component(cid))
            else:
                components.append(layer_state.layer.data.get_component(cid))
        return components


class ScatterLayerState(MatplotlibLayerState):
    """
    A state class that includes all the attributes for layers in a scatter plot.
    """

    size = DDCProperty(docstring="The size of the markers")

    def __init__(self, viewer_state=None, **kwargs):

        super(ScatterLayerState, self).__init__(viewer_state=viewer_state, **kwargs)

        self.size = self.layer.style.markersize

        self._sync_size = keep_in_sync(self, 'size', self.layer.style, 'markersize')
