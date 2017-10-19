# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from glue.core import Data

from glue.config import colormaps
from glue.viewers.matplotlib.state import (MatplotlibDataViewerState,
                                           MatplotlibLayerState,
                                           DeferredDrawCallbackProperty as DDCProperty,
                                           DeferredDrawSelectionCallbackProperty as DDSCProperty)
from glue.core.state_objects import StateAttributeLimitsHelper
from glue.external.echo import keep_in_sync, delay_callback
from glue.core.data_combo_helper import ComponentIDComboHelper
from glue.core.exceptions import IncompatibleAttribute

__all__ = ['ScatterViewerState', 'ScatterLayerState']


class ScatterViewerState(MatplotlibDataViewerState):
    """
    A state class that includes all the attributes for a scatter viewer.
    """

    x_att = DDSCProperty(docstring='The attribute to show on the x-axis', default_index=0)
    y_att = DDSCProperty(docstring='The attribute to show on the y-axis', default_index=1)

    def __init__(self, **kwargs):

        super(ScatterViewerState, self).__init__()

        self.limits_cache = {}

        self.x_lim_helper = StateAttributeLimitsHelper(self, attribute='x_att',
                                                       lower='x_min', upper='x_max',
                                                       log='x_log',
                                                       limits_cache=self.limits_cache)

        self.y_lim_helper = StateAttributeLimitsHelper(self, attribute='y_att',
                                                       lower='y_min', upper='y_max',
                                                       log='y_log',
                                                       limits_cache=self.limits_cache)

        self.add_callback('layers', self._layers_changed)

        self.x_att_helper = ComponentIDComboHelper(self, 'x_att')
        self.y_att_helper = ComponentIDComboHelper(self, 'y_att')

        self.update_from_dict(kwargs)

    def _update_priority(self, name):
        if name == 'layers':
            return 2
        elif name.endswith('_log'):
            return 0.5
        elif name.endswith(('_min', '_max')):
            return 0
        else:
            return 1

    def flip_x(self):
        """
        Flip the x_min/x_max limits.
        """
        self.x_lim_helper.flip_limits()

    def flip_y(self):
        """
        Flip the y_min/y_max limits.
        """
        self.y_lim_helper.flip_limits()

    def _get_x_components(self):
        return self._get_components(self.x_att)

    def _get_y_components(self):
        return self._get_components(self.y_att)

    def _get_components(self, cid):

        # Construct list of components over all layers

        components = []

        for layer_state in self.layers:

            if isinstance(layer_state.layer, Data):
                layer = layer_state.layer
            else:
                layer = layer_state.layer.data

            try:
                components.append(layer.data.get_component(cid))
            except IncompatibleAttribute:
                pass

        return components

    def _layers_changed(self, *args):

        layers_data = self.layers_data
        layers_data_cache = getattr(self, '_layers_data_cache', [])

        if layers_data == layers_data_cache:
            return

        self.x_att_helper.set_multiple_data(self.layers_data)
        self.y_att_helper.set_multiple_data(self.layers_data)

        self._layers_data_cache = layers_data


def display_func_slow(x):
    if x == 'Linear':
        return 'Linear (WARNING: may be slow due to data size)'
    else:
        return x


class ScatterLayerState(MatplotlibLayerState):
    """
    A state class that includes all the attributes for layers in a scatter plot.
    """

    # Color

    cmap_mode = DDSCProperty(docstring="Whether to use color to encode an attribute")
    cmap_att = DDSCProperty(docstring="The attribute to use for the color")
    cmap_vmin = DDCProperty(docstring="The lower level for the colormap")
    cmap_vmax = DDCProperty(docstring="The upper level for the colormap")
    cmap = DDCProperty(docstring="The colormap to use (when in colormap mode)")

    # Markers

    markers_visible = DDCProperty(True, docstring="Whether to show markers")
    size = DDCProperty(docstring="The size of the markers")
    size_mode = DDSCProperty(docstring="Whether to use size to encode an attribute")
    size_att = DDSCProperty(docstring="The attribute to use for the size")
    size_vmin = DDCProperty(docstring="The lower level for the size mapping")
    size_vmax = DDCProperty(docstring="The upper level for the size mapping")
    size_scaling = DDCProperty(1, docstring="Relative scaling of the size")

    # Line

    line_visible = DDCProperty(False, docstring="Whether to show a line connecting all positions")
    linewidth = DDCProperty(1, docstring="The line width")
    linestyle = DDSCProperty(docstring="The line style")

    # Errorbars

    xerr_visible = DDCProperty(False, docstring="Whether to show x error bars")
    yerr_visible = DDCProperty(False, docstring="Whether to show y error bars")
    xerr_att = DDSCProperty(docstring="The attribute to use for the x error bars")
    yerr_att = DDSCProperty(docstring="The attribute to use for the y error bars")

    # Vectors

    vector_visible = DDCProperty(False, docstring="Whether to show vector plot")
    vx_att = DDSCProperty(docstring="The attribute to use for the x vector arrow")
    vy_att = DDSCProperty(docstring="The attribute to use for the y vector arrow")
    vector_arrowhead = DDCProperty(False, docstring="Whether to show vector arrow")
    vector_mode = DDSCProperty(default_index=0, docstring="Which attribute to use for plotting vectors")
    vector_origin = DDSCProperty(default_index=1, docstring="The attribute to use for the arrow position")
    vector_scaling = DDCProperty(1, docstring="The relative scaling of the arrow length")

    def __init__(self, viewer_state=None, layer=None, **kwargs):

        super(ScatterLayerState, self).__init__(viewer_state=viewer_state, layer=layer)

        self.limits_cache = {}

        self.cmap_lim_helper = StateAttributeLimitsHelper(self, attribute='cmap_att',
                                                          lower='cmap_vmin', upper='cmap_vmax',
                                                          limits_cache=self.limits_cache)

        self.size_lim_helper = StateAttributeLimitsHelper(self, attribute='size_att',
                                                          lower='size_vmin', upper='size_vmax',
                                                          limits_cache=self.limits_cache)

        self.cmap_att_helper = ComponentIDComboHelper(self, 'cmap_att',
                                                      numeric=True, categorical=False)

        self.size_att_helper = ComponentIDComboHelper(self, 'size_att',
                                                      numeric=True, categorical=False)

        self.xerr_att_helper = ComponentIDComboHelper(self, 'xerr_att',
                                                      numeric=True, categorical=False)

        self.yerr_att_helper = ComponentIDComboHelper(self, 'yerr_att',
                                                      numeric=True, categorical=False)

        self.vx_att_helper = ComponentIDComboHelper(self, 'vx_att',
                                                    numeric=True, categorical=False)

        self.vy_att_helper = ComponentIDComboHelper(self, 'vy_att',
                                                    numeric=True, categorical=False)

        ScatterLayerState.cmap_mode.set_choices(self, ['Fixed', 'Linear'])
        ScatterLayerState.size_mode.set_choices(self, ['Fixed', 'Linear'])

        linestyle_display = {'solid': '–––––––',
                             'dashed': '– – – – –',
                             'dotted': '· · · · · · · ·',
                             'dashdot': '– · – · – ·'}

        ScatterLayerState.linestyle.set_choices(self, ['solid', 'dashed', 'dotted', 'dashdot'])
        ScatterLayerState.linestyle.set_display_func(self, linestyle_display.get)

        ScatterLayerState.vector_mode.set_choices(self, ['Cartesian', 'Polar'])

        vector_origin_display = {'tail': 'Tail of vector',
                                 'middle': 'Middle of vector',
                                 'tip': 'Tip of vector'}

        ScatterLayerState.vector_origin.set_choices(self, ['tail', 'middle', 'tip'])
        ScatterLayerState.vector_origin.set_display_func(self, vector_origin_display.get)

        self.add_callback('layer', self._on_layer_change)
        if layer is not None:
            self._on_layer_change()

        self.cmap = colormaps.members[0][1]

        self.size = self.layer.style.markersize

        self._sync_size = keep_in_sync(self, 'size', self.layer.style, 'markersize')

        self.update_from_dict(kwargs)

    def _on_layer_change(self, layer=None):

        with delay_callback(self, 'cmap_vmin', 'cmap_vmax', 'size_vmin', 'size_vmax'):

            if self.layer is None:
                self.cmap_att_helper.set_multiple_data([])
                self.size_att_helper.set_multiple_data([])
            else:
                self.cmap_att_helper.set_multiple_data([self.layer])
                self.size_att_helper.set_multiple_data([self.layer])

            if self.layer is None:
                self.xerr_att_helper.set_multiple_data([])
                self.yerr_att_helper.set_multiple_data([])
            else:
                self.xerr_att_helper.set_multiple_data([self.layer])
                self.yerr_att_helper.set_multiple_data([self.layer])

            if self.layer is None:
                self.vx_att_helper.set_multiple_data([])
                self.vy_att_helper.set_multiple_data([])
            else:
                self.vx_att_helper.set_multiple_data([self.layer])
                self.vy_att_helper.set_multiple_data([self.layer])

    def flip_cmap(self):
        """
        Flip the cmap_vmin/cmap_vmax limits.
        """
        self.cmap_lim_helper.flip_limits()

    def flip_size(self):
        """
        Flip the size_vmin/size_vmax limits.
        """
        self.size_lim_helper.flip_limits()
