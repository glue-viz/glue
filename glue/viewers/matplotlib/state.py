from __future__ import absolute_import, division, print_function

import numpy as np

from glue.external.echo import (CallbackProperty, ListCallbackProperty,
                                SelectionCallbackProperty, keep_in_sync,
                                delay_callback)

from glue.core.state_objects import State
from glue.core.message import LayerArtistUpdatedMessage
from glue.core.exceptions import IncompatibleAttribute

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

    def __init__(self, *args, **kwargs):
        super(MatplotlibDataViewerState, self).__init__(*args, **kwargs)
        self.add_callback('x_log', self._on_x_log_update)
        self.add_callback('y_log', self._on_y_log_update)

    def _on_x_log_update(self, x_log):
        self._on_log_update('x')

    def _on_y_log_update(self, y_log):
        self._on_log_update('y')

    def _on_log_update(self, axis):

        # Called when the x_log or y_log setting changes, and we figure out the
        # optimal range to show. The axis argument should be set to 'x' or 'y'

        att = getattr(self, axis + '_att', None)
        min_values, max_values = [], []

        print('_on_log_update', axis, att)

        if att is not None:
            for layer in self.layers:
                if layer.layer is not None:
                    try:
                        data = layer.layer[att]
                    except IncompatibleAttribute:
                        continue
                    if getattr(self, axis + '_log'):
                        data = data[data > 0]
                    else:
                        data = data[~np.isnan(data)]
                    if data.size == 0:
                        continue
                    min_values.append(data.min())
                    max_values.append(data.max())

        min_name = axis + '_min'
        max_name = axis + '_max'

        with delay_callback(self, min_name, max_name):
            if len(min_values) > 0:
                setattr(self, min_name, min(min_values))
                setattr(self, max_name, max(max_values))
            else:
                if getattr(self, max_name) is not None and getattr(self, max_name) < 0:
                    setattr(self, max_name, 10)
                if getattr(self, min_name) is not None and getattr(self, min_name) < 0:
                    setattr(self, min_name, min(0.1, getattr(self, max_name) / 100))

    @property
    def layers_data(self):
        return [layer_state.layer for layer_state in self.layers]


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

        self.add_global_callback(self._notify_layer_update)

    def _notify_layer_update(self, **kwargs):
        message = LayerArtistUpdatedMessage(self)
        if self.layer is not None and self.layer.hub is not None:
            self.layer.hub.broadcast(message)
