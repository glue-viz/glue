from __future__ import absolute_import, division, print_function

from glue.external.echo import CallbackProperty, ListCallbackProperty

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
