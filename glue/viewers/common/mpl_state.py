from __future__ import absolute_import, division, print_function

from glue.external.echo import CallbackProperty, ListCallbackProperty, keep_in_sync

from glue.core.state_objects import State

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

    x_log = DeferredDrawCallbackProperty(False)
    y_log = DeferredDrawCallbackProperty(False)

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

        self._sync_color = keep_in_sync(self, 'color', self.layer.style, 'color')
        self._sync_alpha = keep_in_sync(self, 'alpha', self.layer.style, 'alpha')
