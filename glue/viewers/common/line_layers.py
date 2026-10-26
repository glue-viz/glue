import numpy as np

from echo import ignore_callback

from glue.config import layer_artist_maker
from glue.core import BaseData, Subset
from glue.core.units import UnitConverter
from glue.utils import defer_draw, ensure_numerical, datetime64_to_mpl
from glue.viewers.matplotlib.state import (MatplotlibLayerState,
                                           DeferredDrawCallbackProperty as DDCProperty,
                                           DeferredDrawSelectionCallbackProperty as DDSCProperty)

__all__ = ['LineLayerState', 'BaseLineLayerArtist', 'add_line_layer']


class LineLayerState(MatplotlibLayerState):
    """
    A state class for layers shown as full-height vertical or full-width
    horizontal lines.
    """

    linewidth = DDCProperty(1, docstring='The width of the lines')
    linestyle = DDSCProperty(docstring='The style of the lines')

    def __init__(self, viewer_state=None, layer=None, **kwargs):

        super(LineLayerState, self).__init__(viewer_state=viewer_state, layer=layer)

        linestyle_display = {'solid': '–––––––',
                             'dashed': '– – – – –',
                             'dotted': '· · · · · · · ·',
                             'dashdot': '– · – · – ·'}

        LineLayerState.linestyle.set_choices(self, ['solid', 'dashed', 'dotted', 'dashdot'])
        LineLayerState.linestyle.set_display_func(self, linestyle_display.get)

        self.update_from_dict(kwargs)


class BaseLineLayerArtist:
    """
    A mixin class for layer artists that render the values of one of the
    viewer axis attributes for the layer as lines spanning the full extent
    of the other axis.

    Subclasses should define ``_orientation`` as either ``'vertical'`` or
    ``'horizontal'``, which determines whether the positions are taken from
    the viewer ``x_att`` or ``y_att``.
    """

    _orientation = None

    @classmethod
    def position_att_name(cls):
        return 'x_att' if cls._orientation == 'vertical' else 'y_att'

    @property
    def _position_att(self):
        return getattr(self._viewer_state, self.position_att_name(), None)

    def compute_line_positions(self):
        """
        The unique values of the position attribute for the layer, converted
        to the axis display units where the viewer defines these.
        """
        values = ensure_numerical(self.layer[self._position_att].ravel())
        if values.dtype.kind == 'M':
            values = datetime64_to_mpl(values)
        display_unit = getattr(self._viewer_state,
                               self.position_att_name().replace('att', 'display_unit'), None)
        reference_data = getattr(self._viewer_state, 'reference_data', None)
        if display_unit is not None and reference_data is not None:
            converter = UnitConverter()
            values = converter.to_unit(reference_data, self._position_att,
                                       values, display_unit)
        return np.unique(values[~np.isnan(values)])

    def _changed_position_properties(self, changed):
        # The properties which, when changed, require the line positions to
        # be updated - the display unit property only exists for some viewers.
        att = self.position_att_name()
        return bool({'layer', att, att + '_pixel',
                     att.replace('att', 'display_unit'),
                     'reference_data'} & set(changed))


@defer_draw
def add_line_layer(viewer, layer, artist_cls):
    """
    Add a dataset or subset to a viewer as lines at the values of one of the
    viewer axis attributes, using the specified layer artist class.

    Any existing subsets of a dataset are added along with it, and any
    subsets created afterwards will also be shown as lines.

    Parameters
    ----------
    viewer : `~glue.viewers.common.viewer.Viewer`
        The viewer to add the layer to.
    layer : `~glue.core.data.BaseData` or `~glue.core.subset.Subset`
        The dataset or subset to show as lines.
    artist_cls : type
        The `BaseLineLayerArtist` subclass to use for the layer.

    Returns
    -------
    artist : `BaseLineLayerArtist` instance
        The layer artist that was added to the viewer.
    """
    if not hasattr(viewer.state, artist_cls.position_att_name()):
        raise ValueError("Cannot add {0} lines to this viewer since the viewer "
                         "does not define a {1} attribute".format(
                             artist_cls._orientation,
                             'y axis' if artist_cls._orientation == 'horizontal' else 'x axis'))
    artist = viewer.get_layer_artist(artist_cls, layer=layer)
    # Adding the artist to the container assigns the zorder, which would
    # otherwise trigger a redundant update before the forced one below.
    with ignore_callback(artist.state, 'zorder'):
        viewer._layer_artist_container.append(artist)
    artist.update()
    viewer.draw_legend()
    if isinstance(layer, BaseData):
        for subset in layer.subsets:
            add_line_layer(viewer, subset, artist_cls)
    return artist


@layer_artist_maker('line-layer-subsets')
def line_layer_subset_maker(viewer, layer):
    # New subsets follow the parent dataset: if the parent data is shown as
    # lines in this viewer, subsets of it should be too, using the same
    # layer artist class (which also makes this work for any plotting
    # toolkit, not just Matplotlib).
    if isinstance(layer, Subset):
        for artist in viewer._layer_artist_container:
            if isinstance(artist, BaseLineLayerArtist) and artist.layer is layer.data:
                return viewer.get_layer_artist(type(artist), layer=layer)
