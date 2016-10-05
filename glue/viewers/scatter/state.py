from __future__ import absolute_import, division, print_function

from glue.core import Data
from glue.external.echo import CallbackProperty, add_callback
from glue.utils import nonpartial
from glue.config import colormaps

from glue.core.state_objects import State, StateList, StateAttributeLimitsHelper
from glue.utils import avoid_circular

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

    layers = StateList()


def add_link(instance1, prop1, instance2, prop2):

    p1 = getattr(type(instance1), prop1)
    if not isinstance(p1, CallbackProperty):
        raise TypeError("%s is not a CallbackProperty" % prop1)

    p2 = getattr(type(instance2), prop2)
    if not isinstance(p2, CallbackProperty):
        raise TypeError("%s is not a CallbackProperty" % prop2)

    def update_prop1(value):
        setattr(instance1, prop1, value)

    def update_prop2(value):
        setattr(instance2, prop2, value)

    # Inefficient for now, but does the trick
    add_callback(instance1, prop1, update_prop2)
    # add_callback(instance2, prop2, update_prop1)


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

    def __init__(self, viewer_state=None, **kwargs):

        super(ScatterLayerState, self).__init__(**kwargs)

        self.viewer_state = viewer_state

        add_link(self.viewer_state, 'xatt', self, '_xatt')
        add_link(self.viewer_state, 'yatt', self, '_yatt')

        self.color = self.layer.style.color
        self.alpha = self.layer.style.alpha
        self.size = self.layer.style.markersize

        add_callback(self.layer.style, 'color',
                     nonpartial(self.color_from_layer))

        add_callback(self.layer.style, 'alpha',
                     nonpartial(self.alpha_from_layer))

        add_callback(self.layer.style, 'markersize',
                     nonpartial(self.size_from_layer))

        self.connect('color', nonpartial(self.color_to_layer))
        self.connect('alpha', nonpartial(self.alpha_to_layer))
        self.connect('size', nonpartial(self.size_to_layer))

        if isinstance(self.layer, Data):
            data = self.layer
        else:
            data = self.layer.data

        numeric_components = []
        for cid in data.visible_components:
            comp = data.get_component(cid)
            if comp.numeric:
                numeric_components.append(cid)

        self.helper_size = StateAttributeLimitsHelper(self, data='layer',
                                                      attribute='size_attribute',
                                                      vlo='size_vmin', vhi='size_vmax')

        self.helper_cmap = StateAttributeLimitsHelper(self, data='layer',
                                                      attribute='cmap_attribute',
                                                      vlo='cmap_vmin', vhi='cmap_vmax')

        self.helper_vector_x = StateAttributeLimitsHelper(self, data='layer',
                                                          attribute='vector_x_attribute',
                                                          vlo='vector_x_min', vhi='vector_x_max')

        self.helper_vector_y = StateAttributeLimitsHelper(self, data='layer',
                                                          attribute='vector_y_attribute',
                                                          vlo='vector_y_min', vhi='vector_y_max')

        self.helper_hist_x = StateAttributeLimitsHelper(self, data='layer', attribute='_xatt',
                                                        vlo='h_x_min', vhi='h_x_max')

        self.helper_hist_y = StateAttributeLimitsHelper(self, data='layer', attribute='_yatt',
                                                        vlo='h_y_min', vhi='h_y_max')

        self.cmap_attribute = numeric_components[0], self.layer
        self.size_attribute = numeric_components[0], self.layer
        self.vector_x_attribute = numeric_components[0], self.layer
        self.vector_y_attribute = numeric_components[1], self.layer

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
