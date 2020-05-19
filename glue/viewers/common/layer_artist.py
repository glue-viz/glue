from echo import keep_in_sync, CallbackProperty
from glue.core.layer_artist import LayerArtistBase
from glue.viewers.common.state import LayerState
from glue.core.message import LayerArtistVisibilityMessage

__all__ = ['LayerArtist']


class LayerArtist(LayerArtistBase):

    zorder = CallbackProperty()
    visible = CallbackProperty()

    _layer_state_cls = LayerState

    _python_exporter = None

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

        self._reset_cache()

    def _reset_cache(self):
        self._last_viewer_state = {}
        self._last_layer_state = {}

    def _on_visibility_change(self, *args):
        if self.state.layer is not None and self.state.layer.hub is not None:
            self.state.layer.hub.broadcast(LayerArtistVisibilityMessage(self))

    def __gluestate__(self, context):
        return dict(state=context.id(self.state))

    def pop_changed_properties(self):
        """
        Return the names of properties on the viewer and layer state classes
        that have changed since the last call.

        Note that calling this method updates the underlying cache, so if it is
        called immediately after being called a first time, it will return an
        empty set the second time.
        """

        # Figure out which attributes are different from before. Ideally we
        # shouldn't need this but some methods in layer artists are called
        # multiple times if an attribute is changed due to x_att changing then
        # hist_x_min, hist_x_max, etc. This method is called pop because it
        # returns the list of changed properties and resets the cache.

        changed = set()

        for key, value in self._viewer_state.as_dict().items():
            if value != self._last_viewer_state.get(key, None):
                changed.add(key)

        for key, value in self.state.as_dict().items():
            if value != self._last_layer_state.get(key, None):
                changed.add(key)

        self._last_viewer_state.update(self._viewer_state.as_dict())
        self._last_layer_state.update(self.state.as_dict())

        return changed
