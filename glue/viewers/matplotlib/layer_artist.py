import matplotlib.patches as mpatches


from glue.viewers.matplotlib.state import DeferredDrawCallbackProperty
from glue.core.message import ComputationStartedMessage, ComputationEndedMessage
from glue.viewers.common.layer_artist import LayerArtist

__all__ = ['MatplotlibLayerArtist']


class MatplotlibLayerArtist(LayerArtist):

    zorder = DeferredDrawCallbackProperty()
    visible = DeferredDrawCallbackProperty()

    def __init__(self, axes, viewer_state, layer_state=None, layer=None):

        super(MatplotlibLayerArtist, self).__init__(viewer_state,
                                                    layer_state=layer_state,
                                                    layer=layer)
        self.axes = axes
        self.mpl_artists = []

    def notify_start_computation(self):
        """
        Broadcast a message to indicate that this layer artist has started a
        computation (typically used in conjunction with asynchronous
        operations).
        """
        if self.state.layer is not None and self.state.layer.hub is not None:
            self.state.layer.hub.broadcast(ComputationStartedMessage(self))

    def notify_end_computation(self):
        """
        Broadcast a message to indicate that this layer artist has ended a
        computation (typically used in conjunction with asynchronous
        operations).
        """
        if self.state.layer is not None and self.state.layer.hub is not None:
            self.state.layer.hub.broadcast(ComputationEndedMessage(self))

    def clear(self):
        for artist in self.mpl_artists:
            try:
                artist.set_visible(False)
            except AttributeError:  # can happen for e.g. errorbars
                pass

    def remove(self):
        for artist in self.mpl_artists:
            try:
                artist.remove()
            except ValueError:  # already removed
                pass
            except TypeError:  # can happen for e.g. errorbars
                pass
            except AttributeError:  # can happen for Matplotlib 1.4
                pass
        self.mpl_artists[:] = []
        self._notify_start = None

    def get_layer_color(self):
        return self.state.color

    def get_handle_legend(self):
        # The default legend handle for matplotlib viewer
        if self.enabled and self.state.visible:
            handle = mpatches.Patch(color=self.get_layer_color(), alpha=self.layer.style.alpha)
            return handle, self.layer.label, None
        else:
            return None, None, None

    def redraw(self):
        self.axes.figure.canvas.draw_idle()
