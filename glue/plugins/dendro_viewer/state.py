# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from glue.core import Data
from glue.viewers.matplotlib.state import (MatplotlibDataViewerState,
                                           MatplotlibLayerState,
                                           DeferredDrawCallbackProperty as DDCProperty,
                                           DeferredDrawSelectionCallbackProperty as DDSCProperty)
from glue.core.data_combo_helper import ComponentIDComboHelper

from .dendro_helpers import dendrogram_layout

__all__ = ['DendrogramViewerState', 'DendrogramLayerState']


class Layout(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def xy(self):
        return self.x, self.y


class DendrogramViewerState(MatplotlibDataViewerState):
    """
    A state class that includes all the attributes for a dendrogram viewer.
    """

    height_att = DDSCProperty()
    parent_att = DDSCProperty()
    order_att = DDSCProperty()
    y_log = DDCProperty(False)
    select_substruct = DDCProperty(True)
    reference_data = DDCProperty()

    _layout = DDCProperty()

    def __init__(self, **kwargs):

        super(DendrogramViewerState, self).__init__()

        self.add_callback('layers', self._layers_changed)

        self.height_att_helper = ComponentIDComboHelper(self, 'height_att')
        self.parent_att_helper = ComponentIDComboHelper(self, 'parent_att')
        self.order_att_helper = ComponentIDComboHelper(self, 'order_att')

        self.add_callback('height_att', self._update_layout)
        self.add_callback('parent_att', self._update_layout)
        self.add_callback('order_att', self._update_layout)

        self.add_callback('reference_data', self._on_reference_data_change)

        self.update_from_dict(kwargs)

    def _on_reference_data_change(self, data):

        if self.reference_data is None:
            return

        self.height_att = self.reference_data.find_component_id('height')
        self.parent_att = self.reference_data.find_component_id('parent')
        self.order_att = self.height_att

    def _update_layout(self, att):
        if self.height_att is None or self.parent_att is None or self.order_att is None or self.reference_data is None:
            self._layout = None
        else:
            height = self.reference_data[self.height_att].ravel()
            parent = self.reference_data[self.parent_att].astype(int).ravel()
            order = self.reference_data[self.order_att].ravel()
            x, y = dendrogram_layout(parent, height, order)
            self._layout = Layout(x, y)

    def _update_priority(self, name):
        if name == 'layers':
            return 2
        elif name.endswith('_log'):
            return 0.5
        else:
            return 1

    def _layers_changed(self, *args):

        layers_data = self.layers_data
        layers_data_cache = getattr(self, '_layers_data_cache', [])

        if layers_data == layers_data_cache:
            return

        self.height_att_helper.set_multiple_data(layers_data)
        self.parent_att_helper.set_multiple_data(layers_data)
        self.order_att_helper.set_multiple_data(layers_data)

        for layer in layers_data:
            if isinstance(layer, Data):
                self.reference_data = layer
                break

        self._layers_data_cache = layers_data


class DendrogramLayerState(MatplotlibLayerState):
    """
    A state class that includes all the attributes for layers in a dendrogram plot.
    """
    linewidth = DDCProperty(1, docstring="The line width")
