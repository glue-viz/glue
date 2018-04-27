from __future__ import absolute_import, division, print_function

from qtpy import QtGui

from matplotlib.colors import Colormap

from glue.utils.qt import mpl_to_qt_color, tint_pixmap, cmap2pixmap
from glue.icons import icon_path

__all__ = ['symbol_icon', 'layer_icon', 'layer_artist_icon', 'get_icon', 'POINT_ICONS']

POINT_ICONS = {'o': 'glue_circle_point',
               's': 'glue_box_point',
               '^': 'glue_triangle_up',
               '*': 'glue_star',
               '+': 'glue_cross'}


def symbol_icon(symbol, color=None):
    bm = QtGui.QBitmap(icon_path(POINT_ICONS.get(symbol, 'glue_circle')))

    if color is not None:
        return QtGui.QIcon(tint_pixmap(bm, color))

    return QtGui.QIcon(bm)


def layer_icon(layer):
    """Create a QtGui.QIcon for a Data or Subset instance

    :type layer: :class:`~glue.core.data.Data`,
                 :class:`~glue.core.subset.Subset`,
                 or object with a .style attribute

    :rtype: QtGui.QIcon
    """
    icon = POINT_ICONS.get(layer.style.marker, 'circle_point')
    bm = QtGui.QBitmap(icon_path(icon))
    color = mpl_to_qt_color(layer.style.color)
    pm = tint_pixmap(bm, color)
    return QtGui.QIcon(pm)


def layer_artist_icon(artist):
    """Create a QtGui.QIcon for a LayerArtist instance"""

    # TODO: need a test for this

    from glue.viewers.scatter.layer_artist import ScatterLayerArtist

    color = artist.get_layer_color()

    if isinstance(color, Colormap):
        pm = cmap2pixmap(color)
    else:
        if isinstance(artist, ScatterLayerArtist):
            bm = QtGui.QBitmap(icon_path(POINT_ICONS.get(artist.layer.style.marker,
                                                         'glue_circle_point')))
        else:
            bm = QtGui.QBitmap(icon_path('glue_box_point'))
        color = mpl_to_qt_color(color)
        pm = tint_pixmap(bm, color)

    return QtGui.QIcon(pm)


def get_icon(icon_name):
    """
    Build a QtGui.QIcon from an image name

    Parameters
    ----------
    icon_name : str
      Name of image file. Assumed to be a png file in glue/qt/icons
      Do not include the extension

    Returns
    -------
    A QtGui.QIcon object
    """
    return QtGui.QIcon(icon_path(icon_name))
