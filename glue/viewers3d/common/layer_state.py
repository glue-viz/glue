from echo import CallbackProperty, keep_in_sync
from glue.core.message import LayerArtistUpdatedMessage
from glue.viewers.common.state import LayerState

__all__ = ['LayerState3D']


class LayerState3D(LayerState):
    """
    A base state object for all Vispy layers
    """

    color = CallbackProperty()
    alpha = CallbackProperty()

    def __init__(self, **kwargs):

        super(LayerState3D, self).__init__(**kwargs)

        self._sync_color = None
        self._sync_alpha = None

        self.add_callback('layer', self._layer_changed)
        self._layer_changed()

        self.add_global_callback(self._notify_layer_update)

    def _notify_layer_update(self, **kwargs):
        message = LayerArtistUpdatedMessage(self)
        if self.layer is not None and self.layer.hub is not None:
            self.layer.hub.broadcast(message)

    def _layer_changed(self):

        if self._sync_color is not None:
            self._sync_color.stop_syncing()

        if self._sync_alpha is not None:
            self._sync_alpha.stop_syncing()

        if self.layer is not None:

            self.color = self.layer.style.color
            self.alpha = self.layer.style.alpha

            self._sync_color = keep_in_sync(self, 'color', self.layer.style, 'color')
            self._sync_alpha = keep_in_sync(self, 'alpha', self.layer.style, 'alpha')
