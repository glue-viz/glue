from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib.colors import ColorConverter

from glue.external.qt import QtGui

from matplotlib import cm

__all__ = ['mpl_to_qt4_color', 'qt4_to_mpl_color', 'cmap2pixmap', 'tint_pixmap']


def mpl_to_qt4_color(color, alpha=1.0):
    """
    Convert a matplotlib color stirng into a Qt QColor object

    Parameters
    ----------
    color : str
       A color specification that matplotlib understands
    alpha : float
        Optional opacity. Float in range [0,1]

    Returns
    -------
    qcolor : ``QColor``
        A QColor object representing the converted color
    """
    if color in [None, 'none', 'None']:
        return QtGui.QColor(0, 0, 0, 0)

    cc = ColorConverter()
    r, g, b = cc.to_rgb(color)
    alpha = max(0, min(255, int(256 * alpha)))
    return QtGui.QColor(r * 255, g * 255, b * 255, alpha)


def qt4_to_mpl_color(qcolor):
    """
    Convert a QColor object into a string that matplotlib understands

    Note: This ignores opacity

    Parameters
    ----------
    qcolor : ``QColor``
        The Qt color
        
    Returns
    -------
    color : str
        A hex string describing that color
    """
    hexid = qcolor.name()
    return str(hexid)


def cmap2pixmap(cmap, steps=50):
    """
    Convert a maplotlib colormap into a QPixmap

    Parameters
    ----------
    cmap : `~matplotlib.colors.Colormap`
        The colormap to use
    steps : int
        The number of color steps in the output. Default=50

    Returns
    -------
    pixmap : ``QPixmap``
        The QPixmap instance
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
    """
    Re-color a monochrome pixmap object using `color`

    Parameters
    ----------
    bm : ``QBitmap``
        The Pixmap object
    color : ``QColor``
        The Qt color

    Returns
    -------
    pixmap : ``QPixmap``
        The new pixmap
    """
    if bm.depth() != 1:
        raise TypeError("Input pixmap must have a depth of 1: %i" % bm.depth())

    image = bm.toImage()
    image.setColor(1, color.rgba())
    image.setColor(0, QtGui.QColor(0, 0, 0, 0).rgba())

    result = QtGui.QPixmap.fromImage(image)
    return result
