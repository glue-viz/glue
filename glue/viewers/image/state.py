from __future__ import absolute_import, division, print_function

from glue.external.echo import CallbackProperty, add_callback
from glue.utils import nonpartial
from glue.config import colormaps

from glue.core.state_objects import State, StateList
from glue.utils import avoid_circular

__all__ = ['ImageViewerState', 'ImageLayerState']

FIRST_COLORMAP = colormaps.members[0][1]


class ImageViewerState(State):

    xcoord = CallbackProperty()
    ycoord = CallbackProperty()

    x_min = CallbackProperty()
    x_max = CallbackProperty()

    y_min = CallbackProperty()
    y_max = CallbackProperty()

    aspect = CallbackProperty()

    layers = StateList()


class ImageLayerState(State):

    layer = CallbackProperty()

    att = CallbackProperty()

    color = CallbackProperty()

    alpha = CallbackProperty()

    vmin = CallbackProperty()
    vmax = CallbackProperty()

    def __init__(self, *args, **kwargs):

        super(ImageLayerState, self).__init__(*args, **kwargs)

        self.color = self.layer.style.color
        self.alpha = self.layer.style.alpha

        add_callback(self.layer.style, 'color',
                     nonpartial(self.color_from_layer))

        add_callback(self.layer.style, 'alpha',
                     nonpartial(self.alpha_from_layer))

        self.connect('color', nonpartial(self.color_to_layer))
        self.connect('alpha', nonpartial(self.alpha_to_layer))

    @avoid_circular
    def color_to_layer(self):
        self.layer.style.color = self.color

    @avoid_circular
    def alpha_to_layer(self):
        self.layer.style.alpha = self.alpha

    @avoid_circular
    def color_from_layer(self):
        self.color = self.layer.style.color

    @avoid_circular
    def alpha_from_layer(self):
        self.alpha = self.layer.style.alpha
