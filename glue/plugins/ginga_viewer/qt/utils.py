from __future__ import absolute_import, division, print_function

import numpy as np
from qtpy import QtGui
from glue.core import roi as roimod

__all__ = ['cmap2pixmap', 'ginga_graphic_to_roi']


def cmap2pixmap(cmap, steps=50):
    """Convert a Ginga colormap into a QtGui.QPixmap

    :param cmap: The colormap to use
    :type cmap: Ginga colormap instance (e.g. ginga.cmap.get_cmap('gray'))
    :param steps: The number of color steps in the output. Default=50
    :type steps: int

    :rtype: QtGui.QPixmap
    """
    inds = np.linspace(0, 1, steps)
    n = len(cmap.clst) - 1
    tups = [cmap.clst[int(x * n)] for x in inds]
    rgbas = [QtGui.QColor(int(r * 255), int(g * 255),
                          int(b * 255), 255).rgba() for r, g, b in tups]
    im = QtGui.QImage(steps, 1, QtGui.QImage.Format_Indexed8)
    im.setColorTable(rgbas)
    for i in range(steps):
        im.setPixel(i, 0, i)
    im = im.scaled(128, 32)
    pm = QtGui.QPixmap.fromImage(im)
    return pm


def ginga_graphic_to_roi(obj):
    if obj.kind == 'rectangle':
        roi = roimod.RectangularROI(xmin=obj.x1, xmax=obj.x2,
                                    ymin=obj.y1, ymax=obj.y2)
    elif obj.kind == 'circle':
        roi = roimod.CircularROI(xc=obj.x, yc=obj.y,
                                 radius=obj.radius)
    elif obj.kind == 'polygon':
        vx = map(lambda xy: xy[0], obj.points)
        vy = map(lambda xy: xy[1], obj.points)
        roi = roimod.PolygonalROI(vx=vx, vy=vy)

    else:
        raise Exception("Don't know how to convert shape '%s' to a ROI" % (
            obj.kind))

    return roi
