from __future__ import absolute_import, division, print_function

from glue.core import Data
from glue.external.echo import CallbackProperty, add_callback
from glue.utils import nonpartial
from glue.config import colormaps

from glue.core.state_objects import State, StateList
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


class ScatterLayerState(State):

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

    def __init__(self, *args, **kwargs):

        super(ScatterLayerState, self).__init__(*args, **kwargs)

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

        self.cmap_attribute = numeric_components[0], self.layer
        self.size_attribute = numeric_components[0], self.layer
        self.vector_x_attribute = numeric_components[0], self.layer
        self.vector_y_attribute = numeric_components[1], self.layer

        # FIXME: Shouldn't be needed?
        # self.vector_x_min = self.layer[self.vector_x_attribute[0]].min()
        # self.vector_x_max = self.layer[self.vector_x_attribute[0]].max()
        # self.vector_y_min = self.layer[self.vector_y_attribute[0]].min()
        # self.vector_y_max = self.layer[self.vector_y_attribute[0]].max()


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
