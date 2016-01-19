from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib.colors import ColorConverter

from ...external.qt import QtGui

from matplotlib import cm


def mpl_to_qt4_color(color, alpha=1.0):
    """ Convert a matplotlib color stirng into a Qt QColor object

    :param color:
       A color specification that matplotlib understands
    :type color: str

    :param alpha:
       Optional opacity. Float in range [0,1]
    :type alpha: float

    * Returns *
    A QColor object representing color

    :rtype: QColor
    """
    if color in [None, 'none', 'None']:
        return QtGui.QColor(0, 0, 0, 0)

    cc = ColorConverter()
    r, g, b = cc.to_rgb(color)
    alpha = max(0, min(255, int(256 * alpha)))
    return QtGui.QColor(r * 255, g * 255, b * 255, alpha)


def qt4_to_mpl_color(color):
    """
    Convert a QColor object into a string that matplotlib understands

    Note: This ignores opacity

    :param color: QColor instance

    *Returns*
        A hex string describing that color
    """
    hexid = color.name()
    return str(hexid)


def cmap2pixmap(cmap, steps=50):
    """Convert a maplotlib colormap into a QPixmap

    :param cmap: The colormap to use
    :type cmap: Matplotlib colormap instance (e.g. matplotlib.cm.gray)
    :param steps: The number of color steps in the output. Default=50
    :type steps: int

    :rtype: QPixmap
    """
    sm = cm.ScalarMappable(cmap=cmap)
    sm.norm.vmin = 0.0
    sm.norm.vmax = 1.0
    inds = np.linspace(0, 1, steps)
    rgbas = sm.to_rgba(inds)
    rgbas = [QtGui.QColor(int(r * 255), int(g * 255),
                    int(b * 255), int(a * 255)).rgba() for r, g, b, a in rgbas]
    im = QtGui.QImage(steps, 1, QtGui.QImage.Format_Indexed8)
    im.setColorTable(rgbas)
    for i in range(steps):
        im.setPixel(i, 0, i)
    im = im.scaled(100, 100)
    pm = QtGui.QPixmap.fromImage(im)
    return pm
    
    
def tint_pixmap(bm, color):
    """Re-color a monochrome pixmap object using `color`

    :param bm: QBitmap instance
    :param color: QColor instance

    :rtype: QPixmap. The new pixma;
    """
    if bm.depth() != 1:
        raise TypeError("Input pixmap must have a depth of 1: %i" % bm.depth())

    image = bm.toImage()
    image.setColor(1, color.rgba())
    image.setColor(0, QtGui.QColor(0, 0, 0, 0).rgba())

    result = QtGui.QPixmap.fromImage(image)
    return result
