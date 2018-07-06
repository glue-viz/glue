from __future__ import absolute_import, division, print_function

from glue.external.echo import keep_in_sync, CallbackProperty
from glue.core.layer_artist import LayerArtistBase
from glue.viewers.common.state import LayerState
from glue.core.message import LayerArtistVisibilityMessage

__all__ = ['LayerArtist']


class LayerArtist(LayerArtistBase):

    zorder = CallbackProperty()
    visible = CallbackProperty()

    _layer_state_cls = LayerState

    def __init__(self, viewer_state, layer_state=None, layer=None):

        super(LayerArtist, self).__init__(layer)

        self._viewer_state = viewer_state

        self.layer = layer or layer_state.layer
        self.state = layer_state or self._layer_state_cls(viewer_state=viewer_state,
                                                          layer=self.layer)

        if self.state not in self._viewer_state.layers:
            self._viewer_state.layers.append(self.state)

        self.zorder = self.state.zorder
        self.visible = self.state.visible

        self._sync_zorder = keep_in_sync(self, 'zorder', self.state, 'zorder')
        self._sync_visible = keep_in_sync(self, 'visible', self.state, 'visible')

        self.state.add_callback('visible', self._on_visibility_change)

    def _on_visibility_change(self, *args):
        if self.state.layer is not None and self.state.layer.hub is not None:
            self.state.layer.hub.broadcast(LayerArtistVisibilityMessage(self))

    def __gluestate__(self, context):
        return dict(state=context.id(self.state))
