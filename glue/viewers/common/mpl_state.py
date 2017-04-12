from __future__ import absolute_import, division, print_function

from glue.external.echo import add_callback, CallbackProperty, ListCallbackProperty
from glue.utils import nonpartial

from glue.core.state_objects import State
from glue.utils.decorators import avoid_circular

from glue.utils import defer_draw


class DeferredDrawCallbackProperty(CallbackProperty):
    """
    A callback property where drawing is deferred until
    after notify has called all callback functions.
    """

    @defer_draw
    def notify(self, *args, **kwargs):
        super(DeferredDrawCallbackProperty, self).notify(*args, **kwargs)


class MatplotlibDataViewerState(State):

    x_min = DeferredDrawCallbackProperty()
    x_max = DeferredDrawCallbackProperty()

    y_min = DeferredDrawCallbackProperty()
    y_max = DeferredDrawCallbackProperty()

    log_x = DeferredDrawCallbackProperty(False)
    log_y = DeferredDrawCallbackProperty(False)

    layers = ListCallbackProperty()


class MatplotlibLayerState(State):

    layer = DeferredDrawCallbackProperty()
    color = DeferredDrawCallbackProperty()
    alpha = DeferredDrawCallbackProperty()
    zorder = DeferredDrawCallbackProperty(0)
    visible = DeferredDrawCallbackProperty(True)

    def __init__(self, viewer_state=None, **kwargs):

        super(MatplotlibLayerState, self).__init__(**kwargs)

        self.viewer_state = viewer_state

        self.color = self.layer.style.color
        self.alpha = self.layer.style.alpha

        add_callback(self.layer.style, 'color',
                     nonpartial(self.color_from_layer))

        add_callback(self.layer.style, 'alpha',
                     nonpartial(self.alpha_from_layer))

        # TODO: can we use keep_in_sync here?

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
