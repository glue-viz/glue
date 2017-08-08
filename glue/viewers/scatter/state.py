from __future__ import absolute_import, division, print_function

from glue.core import Data, Subset

from glue.viewers.matplotlib.state import (MatplotlibDataViewerState,
                                           MatplotlibLayerState,
                                           DeferredDrawCallbackProperty as DDCProperty,
                                           DeferredDrawSelectionCallbackProperty as DDSCProperty)
from glue.core.state_objects import StateAttributeLimitsHelper
from glue.external.echo import keep_in_sync
from glue.core.data_combo_helper import ComponentIDComboHelper
from glue.core.exceptions import IncompatibleAttribute

__all__ = ['ScatterViewerState', 'ScatterLayerState']


class ScatterViewerState(MatplotlibDataViewerState):
    """
    A state class that includes all the attributes for a scatter viewer.
    """

    x_att = DDSCProperty(docstring='The attribute to show on the x-axis', default_index=0)
    y_att = DDSCProperty(docstring='The attribute to show on the y-axis', default_index=1)

    def __init__(self, **kwargs):

        super(ScatterViewerState, self).__init__()

        self.limits_cache = {}

        self.x_lim_helper = StateAttributeLimitsHelper(self, attribute='x_att',
                                                       lower='x_min', upper='x_max',
                                                       log='x_log',
                                                       limits_cache=self.limits_cache)

        self.y_lim_helper = StateAttributeLimitsHelper(self, attribute='y_att',
                                                       lower='y_min', upper='y_max',
                                                       log='y_log',
                                                       limits_cache=self.limits_cache)

        self.add_callback('layers', self._layers_changed)

        self.x_att_helper = ComponentIDComboHelper(self, 'x_att')
        self.y_att_helper = ComponentIDComboHelper(self, 'y_att')

        self.update_from_dict(kwargs)

    def _update_priority(self, name):
        if name == 'layers':
            return 2
        elif name.endswith('_log'):
            return 0.5
        elif name.endswith(('_min', '_max')):
            return 0
        else:
            return 1

    def flip_x(self):
        """
        Flip the x_min/x_max limits.
        """
        self.x_lim_helper.flip_limits()

    def flip_y(self):
        """
        Flip the y_min/y_max limits.
        """
        self.y_lim_helper.flip_limits()

    def _get_x_components(self):
        return self._get_components(self.x_att)

    def _get_y_components(self):
        return self._get_components(self.y_att)

    def _get_components(self, cid):

        # Construct list of components over all layers

        components = []

        for layer_state in self.layers:

            if isinstance(layer_state.layer, Data):
                layer = layer_state.layer
            else:
                layer = layer_state.layer.data

            try:
                components.append(layer.data.get_component(cid))
            except IncompatibleAttribute:
                pass

        return components

    def _layers_changed(self, *args):
        self.x_att_helper.set_multiple_data(self.layers_data)
        self.y_att_helper.set_multiple_data(self.layers_data)


class ScatterLayerState(MatplotlibLayerState):
    """
    A state class that includes all the attributes for layers in a scatter plot.
    """

    size = DDCProperty(docstring="The size of the markers")

    def __init__(self, viewer_state=None, **kwargs):

        super(ScatterLayerState, self).__init__(viewer_state=viewer_state, **kwargs)

        self.size = self.layer.style.markersize

        self._sync_size = keep_in_sync(self, 'size', self.layer.style, 'markersize')
