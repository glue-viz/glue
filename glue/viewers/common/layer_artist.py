from __future__ import absolute_import, division, print_function

from glue.external.echo import keep_in_sync, CallbackProperty
from glue.core.layer_artist import LayerArtistBase

__all__ = ['LayerArtistWithState']


class LayerArtistWithState(LayerArtistBase):

    zorder = CallbackProperty()
    visible = CallbackProperty()

    _layer_state_cls = None

    def __init__(self, viewer_state, layer_state=None, layer=None):

        super(LayerArtistWithState, self).__init__(layer)

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

    def __gluestate__(self, context):
        return dict(state=context.id(self.state))
