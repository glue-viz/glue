from ....external.qt import QtGui

from ..colors import qt4_to_mpl_color


def test_colors():
    assert qt4_to_mpl_color(QtGui.QColor(255, 0, 0)) == '#ff0000'
    assert qt4_to_mpl_color(QtGui.QColor(255, 255, 255)) == '#ffffff'


# TODO: add a test for the other way around

# TODO: add a test for cmap2pixmap

# TODO: add a test for tint_pixmap