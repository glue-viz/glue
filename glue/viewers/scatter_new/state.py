from __future__ import absolute_import, division, print_function

from glue.core import Data
from glue.external.echo import CallbackProperty, ListCallbackProperty, add_callback
from glue.utils import nonpartial
from glue.config import colormaps

from glue.core.state_objects import State, StateAttributeLimitsHelper
from glue.utils.decorators import avoid_circular

__all__ = ['ScatterViewerState', 'ScatterLayerState']

FIRST_COLORMAP = colormaps.members[0][1]


class ScatterViewerState(State):

    xatt = CallbackProperty()
    yatt = CallbackProperty()

    x_min = CallbackProperty()
    x_max = CallbackProperty()

    y_min = CallbackProperty()
    y_max = CallbackProperty()

    log_x = CallbackProperty()
    log_y = CallbackProperty()

    layers = ListCallbackProperty()

    def __init__(self, **kwargs):

        super(ScatterViewerState, self).__init__(**kwargs)

        self.limits_cache = {}

        self.xatt_helper = StateAttributeLimitsHelper(self, attribute='xatt',
                                                      lower='x_min', upper='x_max',
                                                      log='log_x',
                                                      limits_cache=self.limits_cache)

        self.yatt_helper = StateAttributeLimitsHelper(self, attribute='yatt',
                                                      lower='y_min', upper='y_max',
                                                      log='log_y',
                                                      limits_cache=self.limits_cache)

    def reset_limits(self):
        self.limits_cache.clear()
        self.xatt_helper._update_limits()
        self.yatt_helper._update_limits()

    def flip_x(self):
        self.xatt_helper.flip_limits()

    def flip_y(self):
        self.yatt_helper.flip_limits()


def add_one_way_link(instance1, prop1, instance2, prop2):
    def update_prop2(value):
        setattr(instance2, prop2, value)
    add_callback(instance1, prop1, update_prop2)


class ScatterLayerState(State):

    # These two will be linked to the reference values in the viewer state
    _xatt = CallbackProperty()
    _yatt = CallbackProperty()

    style = CallbackProperty('Scatter')

    layer = CallbackProperty()

    color_mode = CallbackProperty('Fixed')

    color = CallbackProperty()

    cmap_attribute = CallbackProperty()
    cmap_vmin = CallbackProperty()
    cmap_vmax = CallbackProperty()
    cmap = CallbackProperty(FIRST_COLORMAP)

    alpha = CallbackProperty()

    size_mode = CallbackProperty('Fixed')

    size = CallbackProperty()

    size_attribute = CallbackProperty()
    size_vmin = CallbackProperty()
    size_vmax = CallbackProperty()

    size_scaling = CallbackProperty(1.)

    linewidth = CallbackProperty(1)
    linestyle = CallbackProperty('solid')

    h_nx = CallbackProperty(20)
    h_x_min = CallbackProperty()
    h_x_max = CallbackProperty()
    h_ny = CallbackProperty(20)
    h_y_min = CallbackProperty()
    h_y_max = CallbackProperty()

    vector_x_attribute = CallbackProperty()
    vector_y_attribute = CallbackProperty()
    vector_x_min = CallbackProperty()
    vector_x_max = CallbackProperty()
    vector_y_min = CallbackProperty()
    vector_y_max = CallbackProperty()
    vector_scale = CallbackProperty(1.)

    link_other = CallbackProperty(True)

    def __init__(self, viewer_state=None, **kwargs):

        super(ScatterLayerState, self).__init__(**kwargs)

        self.viewer_state = viewer_state

        add_one_way_link(self.viewer_state, 'xatt', self, '_xatt')
        add_one_way_link(self.viewer_state, 'yatt', self, '_yatt')

        self.color = self.layer.style.color
        self.alpha = self.layer.style.alpha
        self.size = self.layer.style.markersize

        add_callback(self.layer.style, 'color',
                     nonpartial(self.color_from_layer))

        add_callback(self.layer.style, 'alpha',
                     nonpartial(self.alpha_from_layer))

        add_callback(self.layer.style, 'markersize',
                     nonpartial(self.size_from_layer))

        self.add_callback('color', nonpartial(self.color_to_layer))
        self.add_callback('alpha', nonpartial(self.alpha_to_layer))
        self.add_callback('size', nonpartial(self.size_to_layer))

        if isinstance(self.layer, Data):
            data = self.layer
        else:
            data = self.layer.data

        numeric_components = []
        for cid in data.visible_components:
            comp = data.get_component(cid)
            if comp.numeric:
                numeric_components.append(cid)

        self.helper_size = StateAttributeLimitsHelper(self, attribute='size_attribute',
                                                      lower='size_vmin', upper='size_vmax')

        self.helper_cmap = StateAttributeLimitsHelper(self, attribute='cmap_attribute',
                                                      lower='cmap_vmin', upper='cmap_vmax')

        self.helper_vector_x = StateAttributeLimitsHelper(self, attribute='vector_x_attribute',
                                                          lower='vector_x_min', upper='vector_x_max')

        self.helper_vector_y = StateAttributeLimitsHelper(self, attribute='vector_y_attribute',
                                                          lower='vector_y_min', upper='vector_y_max')

        self.helper_hist_x = StateAttributeLimitsHelper(self, attribute='_xatt',
                                                        lower='h_x_min', upper='h_x_max')

        self.helper_hist_y = StateAttributeLimitsHelper(self, attribute='_yatt',
                                                        lower='h_y_min', upper='h_y_max')

        self.cmap_attribute = numeric_components[0]
        self.size_attribute = numeric_components[0]
        self.vector_x_attribute = numeric_components[0]
        self.vector_y_attribute = numeric_components[1]

        self.add_callback('*', self._keep_in_sync, as_kwargs=True)

        self._active_sync = False

    def _keep_in_sync(self, **kwargs):
        if self._active_sync or len(set(kwargs) - set(['style', 'layer', 'color', 'alpha', 'link_other'])) == 0:
            return
        for layer in self.viewer_state.layers:
            layer._active_sync = True
            if layer is not self and layer.link_other:
                for key, value in kwargs.items():
                    print("Modifying", layer, key, value)
                    setattr(layer, key, value)
            layer._active_sync = False

    @avoid_circular
    def color_to_layer(self):
        self.layer.style.color = self.color

    @avoid_circular
    def alpha_to_layer(self):
        self.layer.style.alpha = self.alpha

    @avoid_circular
    def size_to_layer(self):
        self.layer.style.markersize = self.size

    @avoid_circular
    def color_from_layer(self):
        self.color = self.layer.style.color

    @avoid_circular
    def alpha_from_layer(self):
        self.alpha = self.layer.style.alpha

    @avoid_circular
    def size_from_layer(self):
        self.size = self.layer.style.markersize
