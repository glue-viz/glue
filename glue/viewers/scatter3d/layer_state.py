import numpy as np

from glue.config import colormaps
from glue.utils import categorical_ndarray
from echo import CallbackProperty, SelectionCallbackProperty, CallbackPropertyAlias, keep_in_sync, delay_callback
from glue.core.state_objects import StateAttributeLimitsHelper
from glue.core.data_combo_helper import ComponentIDComboHelper
from ..common3d.layer_state import LayerState3D

__all__ = ['ScatterLayerState']


class ScatterLayerState(LayerState3D):
    """
    A state object for scatter layers
    """

    size_mode = SelectionCallbackProperty()
    size = CallbackProperty()
    size_att = SelectionCallbackProperty()
    size_vmin = CallbackProperty()
    size_vmax = CallbackProperty()
    size_scaling = CallbackProperty(1)

    color_mode = SelectionCallbackProperty()
    cmap_att = SelectionCallbackProperty()
    cmap_vmin = CallbackProperty()
    cmap_vmax = CallbackProperty()
    cmap = CallbackProperty()

    xerr_visible = CallbackProperty(False)
    xerr_att = SelectionCallbackProperty()
    yerr_visible = CallbackProperty(False)
    yerr_att = SelectionCallbackProperty()
    zerr_visible = CallbackProperty(False)
    zerr_att = SelectionCallbackProperty()

    vector_visible = CallbackProperty(False)
    vx_att = SelectionCallbackProperty()
    vy_att = SelectionCallbackProperty()
    vz_att = SelectionCallbackProperty()
    vector_scaling = CallbackProperty(1)
    vector_origin = SelectionCallbackProperty(default_index=1)
    vector_arrowhead = CallbackProperty()

    _size_limits_cache = CallbackProperty({})
    _cmap_limits_cache = CallbackProperty({})

    # Aliases for backwards compatibility with old attribute names
    size_attribute = CallbackPropertyAlias('size_att')
    cmap_attribute = CallbackPropertyAlias('cmap_att')
    xerr_attribute = CallbackPropertyAlias('xerr_att')
    yerr_attribute = CallbackPropertyAlias('yerr_att')
    zerr_attribute = CallbackPropertyAlias('zerr_att')
    vx_attribute = CallbackPropertyAlias('vx_att')
    vy_attribute = CallbackPropertyAlias('vy_att')
    vz_attribute = CallbackPropertyAlias('vz_att')

    def __init__(self, layer=None, **kwargs):

        self._sync_markersize = None

        super(ScatterLayerState, self).__init__(layer=layer)

        if self.layer is not None:

            self.color = self.layer.style.color
            self.size = self.layer.style.markersize
            self.alpha = self.layer.style.alpha

        ScatterLayerState.color_mode.set_choices(self, ['Fixed', 'Linear'])
        ScatterLayerState.size_mode.set_choices(self, ['Fixed', 'Linear'])

        self.size_att_helper = ComponentIDComboHelper(self, 'size_att')
        self.cmap_att_helper = ComponentIDComboHelper(self, 'cmap_att')
        self.xerr_att_helper = ComponentIDComboHelper(self, 'xerr_att', categorical=False)
        self.yerr_att_helper = ComponentIDComboHelper(self, 'yerr_att', categorical=False)
        self.zerr_att_helper = ComponentIDComboHelper(self, 'zerr_att', categorical=False)

        self.vx_att_helper = ComponentIDComboHelper(self, 'vx_att', categorical=False)
        self.vy_att_helper = ComponentIDComboHelper(self, 'vy_att', categorical=False)
        self.vz_att_helper = ComponentIDComboHelper(self, 'vz_att', categorical=False)

        self.size_lim_helper = StateAttributeLimitsHelper(self, attribute='size_att',
                                                          lower='size_vmin', upper='size_vmax',
                                                          cache=self._size_limits_cache)

        self.cmap_lim_helper = StateAttributeLimitsHelper(self, attribute='cmap_att',
                                                          lower='cmap_vmin', upper='cmap_vmax',
                                                          cache=self._cmap_limits_cache)

        vector_origin_display = {'tail': 'Tail of vector',
                                 'middle': 'Middle of vector',
                                 'tip': 'Tip of vector'}
        ScatterLayerState.vector_origin.set_choices(self, ['tail', 'middle', 'tip'])
        ScatterLayerState.vector_origin.set_display_func(self, vector_origin_display.get)

        self.add_callback('layer', self._on_layer_change)
        if layer is not None:
            self._on_layer_change()

        self.cmap = colormaps.members[0][1]

        self.update_from_dict(kwargs)

    def _on_layer_change(self, layer=None):

        with delay_callback(self, 'cmap_vmin', 'cmap_vmax', 'size_vmin', 'size_vmax'):
            helpers = [self.size_att_helper, self.cmap_att_helper,
                       self.xerr_att_helper, self.yerr_att_helper, self.zerr_att_helper,
                       self.vx_att_helper, self.vy_att_helper, self.vz_att_helper]
            if self.layer is None:
                for helper in helpers:
                    helper.set_multiple_data([])
            else:
                for helper in helpers:
                    helper.set_multiple_data([self.layer])

    def _update_priority(self, name):
        return 0 if name.endswith(('vmin', 'vmax')) else 1

    def _layer_changed(self):

        super(ScatterLayerState, self)._layer_changed()

        if self._sync_markersize is not None:
            self._sync_markersize.stop_syncing()

        if self.layer is not None:
            self.size = self.layer.style.markersize
            self._sync_markersize = keep_in_sync(self, 'size', self.layer.style, 'markersize')

    def flip_size(self):
        self.size_lim_helper.flip_limits()

    def flip_cmap(self):
        self.cmap_lim_helper.flip_limits()

    @property
    def point_sizes(self):
        if self.size_mode is None:
            return None
        elif self.size_mode == 'Fixed':
            return self.size * self.size_scaling
        else:
            data = self.layer[self.size_att].ravel()
            if isinstance(data, categorical_ndarray):
                data = data.codes
            if self.size_vmax == self.size_vmin:
                size = np.ones(data.shape) * 10
            else:
                size = (20 * (data - self.size_vmin) /
                        (self.size_vmax - self.size_vmin))
            size_data = size * self.size_scaling
            size_data[np.isnan(data)] = 0.
            return size_data

    @property
    def point_colors(self):
        if self.color_mode is None:
            return None
        elif self.color_mode == 'Fixed':
            return self.color
        else:
            data = self.layer[self.cmap_att].ravel()
            if isinstance(data, categorical_ndarray):
                data = data.codes
            if self.cmap_vmax == self.cmap_vmin:
                cmap_data = np.ones(data.shape) * 0.5
            else:
                cmap_data = ((data - self.cmap_vmin) /
                             (self.cmap_vmax - self.cmap_vmin))
            cmap_data = self.cmap(cmap_data)
            cmap_data[:, 3][np.isnan(data)] = 0.
            return cmap_data
