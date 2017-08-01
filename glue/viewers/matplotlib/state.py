from __future__ import absolute_import, division, print_function

from glue.external.echo import (CallbackProperty, ListCallbackProperty,
                                SelectionCallbackProperty, keep_in_sync)

from glue.core.state_objects import State

from glue.utils import defer_draw

__all__ = ['DeferredDrawSelectionCallbackProperty', 'DeferredDrawCallbackProperty',
           'MatplotlibDataViewerState', 'MatplotlibLayerState']


class DeferredDrawCallbackProperty(CallbackProperty):
    """
    A callback property where drawing is deferred until
    after notify has called all callback functions.
    """

    @defer_draw
    def notify(self, *args, **kwargs):
        super(DeferredDrawCallbackProperty, self).notify(*args, **kwargs)


class DeferredDrawSelectionCallbackProperty(SelectionCallbackProperty):
    """
    A callback property where drawing is deferred until
    after notify has called all callback functions.
    """

    @defer_draw
    def notify(self, *args, **kwargs):
        super(DeferredDrawSelectionCallbackProperty, self).notify(*args, **kwargs)


class MatplotlibDataViewerState(State):
    """
    A base class that includes common attributes for viewers based on
    Matplotlib.
    """

    x_min = DeferredDrawCallbackProperty(docstring='Lower limit of the visible x range')
    x_max = DeferredDrawCallbackProperty(docstring='Upper limit of the visible x range')

    y_min = DeferredDrawCallbackProperty(docstring='Lower limit of the visible y range')
    y_max = DeferredDrawCallbackProperty(docstring='Upper limit of the visible y range')

    x_log = DeferredDrawCallbackProperty(False, docstring='Whether the x axis is logarithmic')
    y_log = DeferredDrawCallbackProperty(False, docstring='Whether the y axis is logarithmic')

    aspect = DeferredDrawCallbackProperty('auto', docstring='Aspect ratio for the axes')

    layers = ListCallbackProperty(docstring='A collection of all layers in the viewer')


class MatplotlibLayerState(State):
    """
    A base class that includes common attributes for all layers in viewers based
    on Matplotlib.
    """

    layer = DeferredDrawCallbackProperty(docstring='The :class:`~glue.core.data.Data` '
                                                   'or :class:`~glue.core.subset.Subset` '
                                                   'represented by the layer')
    color = DeferredDrawCallbackProperty(docstring='The color used to display '
                                                   'the data')
    alpha = DeferredDrawCallbackProperty(docstring='The transparency used to '
                                                   'display the data')
    zorder = DeferredDrawCallbackProperty(0, docstring='A value used to indicate '
                                                       'which layers are shown in '
                                                       'front of which (larger '
                                                       'zorder values are on top '
                                                       'of other layers)')
    visible = DeferredDrawCallbackProperty(True, docstring='Whether the layer '
                                                           'is currently visible')

    def __init__(self, viewer_state=None, **kwargs):

        super(MatplotlibLayerState, self).__init__(**kwargs)

        self.viewer_state = viewer_state

        self.color = self.layer.style.color
        self.alpha = self.layer.style.alpha

        self._sync_color = keep_in_sync(self, 'color', self.layer.style, 'color')
        self._sync_alpha = keep_in_sync(self, 'alpha', self.layer.style, 'alpha')
