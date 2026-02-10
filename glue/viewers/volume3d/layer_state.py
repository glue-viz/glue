from glue.config import colormaps
from glue.core import Subset
from echo import (CallbackProperty, SelectionCallbackProperty,
                  CallbackPropertyAlias, delay_callback)
from glue.core.state_objects import StateAttributeLimitsHelper
from glue.core.data_combo_helper import ComponentIDComboHelper
from glue.viewers.common.stretch_state_mixin import StretchStateMixin
from ..common3d.layer_state import LayerState3D

__all__ = ['VolumeLayerState']


class VolumeLayerState(LayerState3D, StretchStateMixin):
    """
    A state object for volume layers
    """

    attribute = SelectionCallbackProperty()
    v_min = CallbackProperty()
    v_max = CallbackProperty()
    cmap_mode = SelectionCallbackProperty()
    cmap = CallbackProperty()
    subset_mode = CallbackProperty('data')
    _limits_cache = CallbackProperty({})

    # Aliases for backwards compatibility with old attribute names
    vmin = CallbackPropertyAlias('v_min')
    vmax = CallbackPropertyAlias('v_max')
    color_mode = CallbackPropertyAlias('cmap_mode')

    def __init__(self, layer=None, **kwargs):

        super(VolumeLayerState, self).__init__(layer=layer)

        if self.layer is not None:

            self.color = self.layer.style.color
            self.alpha = self.layer.style.alpha

        self.att_helper = ComponentIDComboHelper(self, 'attribute')

        self.lim_helper = StateAttributeLimitsHelper(self, attribute='attribute',
                                                     lower='v_min', upper='v_max',
                                                     cache=self._limits_cache)

        VolumeLayerState.cmap_mode.set_choices(self, ['Fixed', 'Linear'])

        self.setup_stretch_callback()

        self.add_callback('layer', self._on_layer_change)
        if layer is not None:
            self._on_layer_change()

        self.cmap = colormaps.members[0][1]

        if isinstance(self.layer, Subset):
            self.v_min = 0
            self.v_max = 1

        self.update_from_dict(kwargs)

    def _on_layer_change(self, layer=None):

        with delay_callback(self, 'v_min', 'v_min'):

            if self.layer is None:
                self.att_helper.set_multiple_data([])
            else:
                self.att_helper.set_multiple_data([self.layer])

    def _update_priority(self, name):
        return 0 if name.endswith(('v_min', 'v_max')) else 1

    def flip_limits(self):
        self.lim_helper.flip_limits()
