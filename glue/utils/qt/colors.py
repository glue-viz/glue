from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib.colors import ColorConverter

from glue.external.qt import QtCore, QtGui
from glue.external.echo import add_callback
from glue.utils import nonpartial
from glue.utils.qt.helpers import CUSTOM_QWIDGETS

from matplotlib import cm

__all__ = ['mpl_to_qt4_color', 'qt4_to_mpl_color', 'cmap2pixmap',
           'tint_pixmap', 'QColorBox', 'connect_color']


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


def connect_color(client, prop, widget):

    def update_widget(text):
        widget.setColor(text)

    def update_prop():
        setattr(client, prop, widget.color())

    add_callback(client, prop, update_widget)
    widget.colorChanged.connect(update_prop)


class QColorBox(QtGui.QLabel):

    mousePressed = QtCore.Signal()
    colorChanged = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super(QColorBox, self).__init__(*args, **kwargs)
        self.mousePressed.connect(nonpartial(self.query_color))
        self.colorChanged.connect(nonpartial(self.on_color_change))
        self.setColor("#000000")

    def mousePressEvent(self, event):
        self.mousePressed.emit()
        event.accept()

    def query_color(self):
        color = QtGui.QColorDialog.getColor(self._qcolor, parent=self)
        if color.isValid():
            self.setColor(qt4_to_mpl_color(color))

    def setColor(self, color):
        self._color = color
        self.colorChanged.emit()

    def color(self):
        return self._color

    def on_color_change(self):
        self._qcolor = mpl_to_qt4_color(self.color())
        image = QtGui.QImage(70, 22, QtGui.QImage.Format_RGB32)
        image.fill(self._qcolor)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.setPixmap(pixmap)

CUSTOM_QWIDGETS.append(QColorBox)


if __name__ == "__main__":

    from glue.external.qt import get_qapp

    app = get_qapp()

    label = QColorBox()
    label.resize(100,100)
    label.show()
    label.raise_()
    app.exec_()
