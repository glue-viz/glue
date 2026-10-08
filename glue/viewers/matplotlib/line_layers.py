import numpy as np

from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

from glue.core import BaseData
from glue.core.exceptions import IncompatibleAttribute
from glue.utils import defer_draw
from glue.viewers.common.line_layers import LineLayerState, BaseLineLayerArtist, add_line_layer
from glue.viewers.common.python_export import code, serialize_options
from glue.viewers.matplotlib.layer_artist import MatplotlibLayerArtist

__all__ = ['VerticalLineLayerArtist', 'HorizontalLineLayerArtist',
           'add_vertical_lines', 'add_horizontal_lines']


def values_to_segments(values, horizontal=False):
    """
    Construct segments for a `~matplotlib.collections.LineCollection` with one
    full-height vertical (or full-width horizontal) line per value, where the
    direction along the lines is in axes fraction coordinates (0 to 1).
    """
    segments = np.zeros((len(values), 2, 2))
    along, across = (0, 1) if horizontal else (1, 0)
    segments[:, 0, across] = values
    segments[:, 1, across] = values
    segments[:, 1, along] = 1
    return segments


def python_export_line_layer(layer, *args):

    if len(layer.mpl_artists) == 0 or not layer.enabled or not layer.visible:
        return [], None

    horizontal = layer._orientation == 'horizontal'

    options = dict(colors=layer.state.color,
                   linewidth=layer.state.linewidth,
                   linestyle=layer.state.linestyle,
                   alpha=layer.state.alpha,
                   zorder=layer.state.zorder,
                   transform=code('ax.get_yaxis_transform()' if horizontal
                                  else 'ax.get_xaxis_transform()'))

    script = ""
    imports = ["import numpy as np"]

    script += "# Plot lines at the positions along the {0} axis\n".format('y' if horizontal else 'x')
    script += "positions = np.unique(layer_data['{0}'])\n".format(layer._position_att.label)
    script += "positions = positions[~np.isnan(positions)]\n"
    if horizontal:
        script += "line_artist = ax.hlines(positions, 0, 1, {0})\n".format(serialize_options(options))
    else:
        script += "line_artist = ax.vlines(positions, 0, 1, {0})\n".format(serialize_options(options))
    script += "legend_handles.append(line_artist)\n"
    script += "legend_labels.append(layer_data.label)\n"

    return imports, script.strip()


class MatplotlibLineLayerArtist(MatplotlibLayerArtist, BaseLineLayerArtist):
    """
    A layer artist that renders the values of one of the viewer axis
    attributes for the layer as lines spanning the full extent of the other
    axis.

    All the lines are rendered as a single
    `~matplotlib.collections.LineCollection` so that this stays efficient
    even for many thousands of lines.
    """

    _layer_state_cls = LineLayerState
    _python_exporter = python_export_line_layer

    def __init__(self, axes, viewer_state, layer_state=None, layer=None):

        super(MatplotlibLineLayerArtist, self).__init__(axes, viewer_state,
                                                        layer_state=layer_state,
                                                        layer=layer)

        self._viewer_state.add_global_callback(self._update_line_layer)
        self.state.add_global_callback(self._update_line_layer)

        # The line collection uses a blended transform so that the lines
        # always span the full extent of the axes in the direction along the
        # lines, regardless of the limits.
        if self._orientation == 'horizontal':
            transform = self.axes.get_yaxis_transform()
        else:
            transform = self.axes.get_xaxis_transform()
        self.line_collection = LineCollection(np.zeros((0, 2, 2)), transform=transform)
        self.axes.add_collection(self.line_collection)

        self.mpl_artists = [self.line_collection]

    @defer_draw
    def _update_data(self):

        try:
            positions = self.compute_line_positions()
        except (IncompatibleAttribute, IndexError):
            self.line_collection.set_segments(np.zeros((0, 2, 2)))
            self.redraw()
            if isinstance(self.state.layer, BaseData):
                self.disable_invalid_attributes(self._position_att)
            else:
                self.disable_incompatible_subset()
            return

        self.enable()
        # Antialiasing is a significant fraction of the rendering cost for
        # very large numbers of lines and makes no visible difference when
        # the lines are that dense, so we turn it off in that case.
        self.line_collection.set_antialiased(len(positions) < 10000)
        self.line_collection.set_segments(
            values_to_segments(positions, horizontal=self._orientation == 'horizontal'))
        self.redraw()

    @defer_draw
    def _update_visual_attributes(self):

        if not self.enabled:
            return

        self.line_collection.set_visible(self.state.visible)
        self.line_collection.set_zorder(self.state.zorder)
        self.line_collection.set_color(self.state.color)
        self.line_collection.set_alpha(self.state.alpha)
        self.line_collection.set_linewidth(self.state.linewidth)
        self.line_collection.set_linestyle(self.state.linestyle)

        self.redraw()

    def _update_line_layer(self, force=False, **kwargs):

        if self._position_att is None or self.state.layer is None:
            return

        # NOTE: we need to evaluate this even if force=True so that the cache
        # of updated properties is up to date after this method has been called.
        changed = self.pop_changed_properties()

        if force or self._changed_position_properties(changed):
            self._update_data()
            force = True

        if force or any(prop in changed for prop in ('alpha', 'color', 'zorder', 'visible',
                                                     'linewidth', 'linestyle')):
            self._update_visual_attributes()

    @defer_draw
    def update(self):
        self._update_line_layer(force=True)
        self.redraw()

    def get_handle_legend(self):
        if self.enabled and self.state.visible:
            handle = Line2D([0], [0], alpha=self.state.alpha,
                            linestyle=self.state.linestyle,
                            linewidth=self.state.linewidth,
                            color=self.get_layer_color())
            return handle, self.layer.label, None
        else:
            return None, None, None


class VerticalLineLayerArtist(MatplotlibLineLayerArtist):
    """
    A layer artist that renders the values of the viewer x attribute for the
    layer as full-height vertical lines, for use in any Matplotlib viewer
    whose viewer state has an ``x_att`` attribute, e.g. the scatter, profile,
    histogram, and image viewers.
    """

    _orientation = 'vertical'


class HorizontalLineLayerArtist(MatplotlibLineLayerArtist):
    """
    A layer artist that renders the values of the viewer y attribute for the
    layer as full-width horizontal lines, for use in any Matplotlib viewer
    whose viewer state has a ``y_att`` attribute, e.g. the scatter and image
    viewers.
    """

    _orientation = 'horizontal'


def add_vertical_lines(viewer, layer):
    """
    Add a dataset or subset to a Matplotlib-based viewer as full-height
    vertical lines at the values of the viewer x attribute.

    Any existing subsets of a dataset are added along with it, and any
    subsets created afterwards will also be shown as vertical lines.
    """
    return add_line_layer(viewer, layer, VerticalLineLayerArtist)


def add_horizontal_lines(viewer, layer):
    """
    Add a dataset or subset to a Matplotlib-based viewer as full-width
    horizontal lines at the values of the viewer y attribute.

    Any existing subsets of a dataset are added along with it, and any
    subsets created afterwards will also be shown as horizontal lines.
    """
    return add_line_layer(viewer, layer, HorizontalLineLayerArtist)
