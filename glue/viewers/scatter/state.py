# -*- coding: utf-8 -*-

import numpy as np

from glue.core import BaseData, Subset

from glue.config import colormaps
from glue.viewers.matplotlib.state import (MatplotlibDataViewerState,
                                           MatplotlibLayerState,
                                           DeferredDrawCallbackProperty as DDCProperty,
                                           DeferredDrawSelectionCallbackProperty as DDSCProperty)
from glue.core.state_objects import StateAttributeLimitsHelper
from echo import keep_in_sync, delay_callback
from glue.core.data_combo_helper import ComponentIDComboHelper, ComboHelper
from glue.core.exceptions import IncompatibleAttribute

from matplotlib.projections import get_projection_names

__all__ = ['ScatterViewerState', 'ScatterLayerState']


class ScatterViewerState(MatplotlibDataViewerState):
    """
    A state class that includes all the attributes for a scatter viewer.
    """

    x_att = DDSCProperty(docstring='The attribute to show on the x-axis', default_index=0)
    y_att = DDSCProperty(docstring='The attribute to show on the y-axis', default_index=1)
    dpi = DDCProperty(72, docstring='The resolution (in dots per inch) of density maps, if present')
    plot_mode = DDSCProperty(docstring="Whether to plot the data in cartesian, polar or another projection")
    angle_unit = DDSCProperty(docstring="Whether to use radians or degrees for any angular coordinates")

    def __init__(self, **kwargs):

        super(ScatterViewerState, self).__init__()

        self.limits_cache = {}

        self.x_lim_helper = StateAttributeLimitsHelper(self, attribute='x_att',
                                                       lower='x_min', upper='x_max',
                                                       log='x_log', margin=0.04,
                                                       limits_cache=self.limits_cache)

        self.y_lim_helper = StateAttributeLimitsHelper(self, attribute='y_att',
                                                       lower='y_min', upper='y_max',
                                                       log='y_log', margin=0.04,
                                                       limits_cache=self.limits_cache)

        self.add_callback('layers', self._layers_changed)

        self.x_att_helper = ComponentIDComboHelper(self, 'x_att', pixel_coord=True, world_coord=True)
        self.y_att_helper = ComponentIDComboHelper(self, 'y_att', pixel_coord=True, world_coord=True)

        self.plot_mode_helper = ComboHelper(self, 'plot_mode')
        self.plot_mode_helper.choices = [proj for proj in get_projection_names() if proj not in ['3d', 'scatter_density']]
        self.plot_mode_helper.selection = 'rectilinear'

        self.angle_unit_helper = ComboHelper(self, 'angle_unit')
        self.angle_unit_helper.choices = ['radians', 'degrees']
        self.angle_unit_helper.selection = 'radians'

        self.update_from_dict(kwargs)

        self.add_callback('x_log', self._reset_x_limits)
        self.add_callback('y_log', self._reset_y_limits)

        if self.using_polar:
            self.full_circle()

    def _reset_x_limits(self, *args):
        if self.x_att is None:
            return
        self.x_lim_helper.percentile = 100
        self.x_lim_helper.update_values(force=True)

    def _reset_y_limits(self, *args):
        if self.y_att is None:
            return
        self.y_lim_helper.percentile = 100
        self.y_lim_helper.update_values(force=True)

    def reset_limits(self):
        if not self.using_polar:
            self._reset_x_limits()
        self._reset_y_limits()

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

    @property
    def using_rectilinear(self):
        return self.plot_mode == 'rectilinear'

    @property
    def using_polar(self):
        return self.plot_mode == 'polar'

    @property
    def using_full_sphere(self):
        return self.plot_mode in ['aitoff', 'hammer', 'mollweide', 'lambert']

    @property
    def using_degrees(self):
        return (self.using_polar or self.using_full_sphere) and self.angle_unit == 'degrees'

    @property
    def using_radians(self):
        return not self.using_rectilinear and self.angle_unit == 'radians'

    def full_circle(self):
        if not self.using_polar:
            return
        self.x_min = 0
        self.x_max = 2 * np.pi

    @property
    def x_categories(self):
        return self._categories(self.x_att)

    @property
    def y_categories(self):
        return self._categories(self.y_att)

    def _categories(self, cid):

        categories = []

        for layer_state in self.layers:

            if isinstance(layer_state.layer, BaseData):
                layer = layer_state.layer
            else:
                layer = layer_state.layer.data

            try:
                if layer.data.get_kind(cid) == 'categorical':
                    categories.append(layer.data.get_data(cid).categories)
            except IncompatibleAttribute:
                pass

        if len(categories) == 0:
            return None
        else:
            return np.unique(np.hstack(categories))

    @property
    def x_kinds(self):
        return self._component_kinds(self.x_att)

    @property
    def y_kinds(self):
        return self._component_kinds(self.y_att)

    def _component_kinds(self, cid):

        # Construct list of component kinds over all layers

        kinds = set()

        for layer_state in self.layers:

            if isinstance(layer_state.layer, BaseData):
                layer = layer_state.layer
            else:
                layer = layer_state.layer.data

            try:
                kinds.add(layer.data.get_kind(cid))
            except IncompatibleAttribute:
                pass

        return kinds

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

    # Points

    points_mode = DDSCProperty(docstring='Whether to use markers or a density map')

    # Markers

    markers_visible = DDCProperty(True, docstring="Whether to show markers")
    size = DDCProperty(docstring="The size of the markers")
    size_mode = DDSCProperty(docstring="Whether to use size to encode an attribute")
    size_att = DDSCProperty(docstring="The attribute to use for the size")
    size_vmin = DDCProperty(docstring="The lower level for the size mapping")
    size_vmax = DDCProperty(docstring="The upper level for the size mapping")
    size_scaling = DDCProperty(1, docstring="Relative scaling of the size")
    fill = DDCProperty(True, docstring="Whether to fill the markers")

    # Density map

    density_map = DDCProperty(False, docstring="Whether to show the points as a density map")
    stretch = DDSCProperty(default='log', docstring='The stretch used to render the layer, '
                                                    'which should be one of ``linear``, '
                                                    '``sqrt``, ``log``, or ``arcsinh``')
    density_contrast = DDCProperty(1, docstring="The dynamic range of the density map")

    # Note that we keep the dpi in the viewer state since we want it to always
    # be in sync between layers.

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
    vector_mode = DDSCProperty(default_index=0, docstring="Whether to plot the vectors in cartesian or polar mode")
    vector_origin = DDSCProperty(default_index=1, docstring="Whether to place the vector so that the origin is at the tail, middle, or tip")
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
                                                      numeric=True, datetime=False, categorical=False)

        self.size_att_helper = ComponentIDComboHelper(self, 'size_att',
                                                      numeric=True, datetime=False, categorical=False)

        self.xerr_att_helper = ComponentIDComboHelper(self, 'xerr_att',
                                                      numeric=True, datetime=False, categorical=False)

        self.yerr_att_helper = ComponentIDComboHelper(self, 'yerr_att',
                                                      numeric=True, datetime=False, categorical=False)

        self.vx_att_helper = ComponentIDComboHelper(self, 'vx_att',
                                                    numeric=True, datetime=False, categorical=False)

        self.vy_att_helper = ComponentIDComboHelper(self, 'vy_att',
                                                    numeric=True, datetime=False, categorical=False)

        self.points_mode_helper = ComboHelper(self, 'points_mode')

        points_mode_display = {'auto': 'Density map or markers (auto)',
                               'markers': 'Markers',
                               'density': 'Density map'}

        ScatterLayerState.points_mode.set_choices(self, ['auto', 'markers', 'density'])
        ScatterLayerState.points_mode.set_display_func(self, points_mode_display.get)

        self.add_callback('points_mode', self._update_density_map_mode)
        self.add_callback('density_map', self._on_density_map_change, priority=10000)

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

        stretch_display = {'linear': 'Linear',
                           'sqrt': 'Square Root',
                           'arcsinh': 'Arcsinh',
                           'log': 'Logarithmic'}

        ScatterLayerState.stretch.set_choices(self, ['linear', 'sqrt', 'arcsinh', 'log'])
        ScatterLayerState.stretch.set_display_func(self, stretch_display.get)

        if self.viewer_state is not None:
            self.viewer_state.add_callback('x_att', self._on_xy_change, priority=10000)
            self.viewer_state.add_callback('y_att', self._on_xy_change, priority=10000)
            if hasattr(self.viewer_state, 'plot_mode'):
                self.viewer_state.add_callback('plot_mode', self._update_points_mode, priority=10000)
            self._on_xy_change()
            self._update_points_mode()

        self.add_callback('layer', self._on_layer_change)
        if layer is not None:
            self._on_layer_change()

        self.cmap = colormaps.members[0][1]

        self.size = self.layer.style.markersize

        self._sync_size = keep_in_sync(self, 'size', self.layer.style, 'markersize')

        self.update_from_dict(kwargs)

    def _update_points_mode(self, *args):
        if getattr(self.viewer_state, 'using_polar', False) or getattr(self.viewer_state, 'using_full_sphere', False):
            self.points_mode_helper.choices = ['markers']
            self.points_mode_helper.select = 'markers'
        else:
            self.points_mode_helper.choices = ['auto', 'markers', 'density']

    def _on_xy_change(self, *event):

        if self.viewer_state.x_att is None or self.viewer_state.y_att is None:
            return

        if isinstance(self.layer, BaseData):
            layer = self.layer
        else:
            layer = self.layer.data

        try:
            x_datetime = layer.get_kind(self.viewer_state.x_att) == 'datetime'
        except IncompatibleAttribute:
            x_datetime = False

        try:
            y_datetime = layer.get_kind(self.viewer_state.y_att) == 'datetime'
        except IncompatibleAttribute:
            y_datetime = False

        with delay_callback(self, 'xerr_visible', 'yerr_visible', 'vector_visible'):
            if x_datetime:
                self.xerr_visible = False
            if y_datetime:
                self.yerr_visible = False
            if x_datetime or y_datetime:
                self.vector_visible = False

    def _on_layer_change(self, layer=None):

        with delay_callback(self, 'cmap_vmin', 'cmap_vmax', 'size_vmin', 'size_vmax', 'density_map'):

            self._update_density_map_mode()

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

    def _update_density_map_mode(self, *args):
        if self.points_mode == 'auto':
            if self.layer.size > 100000:
                self.density_map = True
            else:
                self.density_map = False
        elif self.points_mode == 'density':
            self.density_map = True
        else:
            self.density_map = False

    def _on_density_map_change(self, *args):
        # If the density map mode is used, we should disable the lines/errors/vectors
        if self.density_map:
            with delay_callback(self,
                                'line_visible', 'xerr_visible',
                                'yerr_visible', 'vector_visible'):
                if self.line_visible:
                    self.line_visible = False
                if self.xerr_visible:
                    self.xerr_visible = False
                if self.yerr_visible:
                    self.yerr_visible = False
                if self.vector_visible:
                    self.vector_visible = False

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

    @property
    def cmap_name(self):
        return colormaps.name_from_cmap(self.cmap)

    def compute_density_map(self, bins=None, range=None):

        if not self.markers_visible or not self.density_map:
            return np.zeros(bins)

        if isinstance(self.layer, Subset):
            data = self.layer.data
            subset_state = self.layer.subset_state
        else:
            data = self.layer
            subset_state = None

        count = data.compute_histogram([self.viewer_state.y_att, self.viewer_state.x_att],
                                        subset_state=subset_state, bins=bins,
                                        log=(self.viewer_state.y_log, self.viewer_state.x_log),
                                        range=range)

        if self.cmap_mode == 'Fixed':
            return count
        else:
            total = data.compute_histogram([self.viewer_state.y_att, self.viewer_state.x_att],
                                            subset_state=subset_state, bins=bins,
                                            weights=self.cmap_att,
                                            log=(self.viewer_state.y_log, self.viewer_state.x_log),
                                            range=range)
            return total / count

    @classmethod
    def __setgluestate__(cls, rec, context):
        # Patch for glue files produced with glue v0.11
        if 'style' in rec['values']:
            style = context.object(rec['values'].pop('style'))
            if style == 'Scatter':
                rec['values']['markers_visible'] = True
                rec['values']['line_visible'] = False
            elif style == 'Line':
                rec['values']['markers_visible'] = False
                rec['values']['line_visible'] = True
        return super(ScatterLayerState, cls).__setgluestate__(rec, context)
