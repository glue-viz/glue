from __future__ import absolute_import, division, print_function

from glue.external.echo import CallbackProperty, ListCallbackProperty, add_callback, keep_in_sync, ignore_callback
from glue.utils import nonpartial
from glue.config import colormaps

from glue.core.state_objects import State, StateAttributeLimitsHelper
from glue.utils.decorators import avoid_circular

__all__ = ['HistogramViewerState', 'HistogramLayerState']


class HistogramViewerState(State):

    xatt = CallbackProperty()
    yatt = CallbackProperty()

    x_min = CallbackProperty()
    x_max = CallbackProperty()

    y_min = CallbackProperty()
    y_max = CallbackProperty()

    log_x = CallbackProperty()
    log_y = CallbackProperty()

    cumulative = CallbackProperty(False)
    normalize = CallbackProperty(False)

    layers = ListCallbackProperty()

    hist_x_min = CallbackProperty()
    hist_x_max = CallbackProperty()
    hist_n_bin = CallbackProperty(10)

    def __init__(self):
        super(HistogramViewerState, self).__init__()
        self.x_att_helper = StateAttributeLimitsHelper(self, attribute='xatt',
                                                       lower='x_min', upper='x_max', log='log_x')
        self.hist_x_att_helper = StateAttributeLimitsHelper(self, attribute='xatt',
                                                            lower='hist_x_min', upper='hist_x_max')

    def flip_x(self):
        self.x_att_helper.flip_limits()


class HistogramLayerState(State):

    layer = CallbackProperty()
    color = CallbackProperty()
    alpha = CallbackProperty()

    def __init__(self, viewer_state=None, **kwargs):

        super(HistogramLayerState, self).__init__(**kwargs)

        self.viewer_state = viewer_state

        self.color = self.layer.style.color
        self.alpha = self.layer.style.alpha

        add_callback(self.layer.style, 'color',
                     nonpartial(self.color_from_layer))

        add_callback(self.layer.style, 'alpha',
                     nonpartial(self.alpha_from_layer))

        self.add_callback('color', nonpartial(self.color_to_layer))
        self.add_callback('alpha', nonpartial(self.alpha_to_layer))

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
