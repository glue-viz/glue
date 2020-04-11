from echo import CallbackProperty, ListCallbackProperty

from glue.core.state_objects import State

__all__ = ['ViewerState', 'LayerState']


class ViewerState(State):
    """
    A base class for all viewer states.
    """

    layers = ListCallbackProperty(docstring='A collection of all layers in the viewer')

    @property
    def layers_data(self):
        return [layer_state.layer for layer_state in self.layers]


class LayerState(State):
    """
    A base class for all layer states.
    """

    layer = CallbackProperty(docstring='The :class:`~glue.core.data.Data` or '
                                       ':class:`~glue.core.subset.Subset` '
                                       'represented by the layer')
    zorder = CallbackProperty(0, docstring='A value used to indicate which '
                                           'layers are shown in front of which '
                                           '(larger zorder values are on top of '
                                           'other layers)')
    visible = CallbackProperty(True, docstring='Whether the layer is currently visible')

    def __init__(self, viewer_state=None, **kwargs):
        super(LayerState, self).__init__(**kwargs)
        self.viewer_state = viewer_state

    def __repr__(self):
        if self.layer is None:
            return "%s with layer unset" % (self.__class__.__name__)
        else:
            return "%s for %s" % (self.__class__.__name__, self.layer.label)
