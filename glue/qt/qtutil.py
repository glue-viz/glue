"""
Various standalone utility code for
working with Qt
"""

from __future__ import absolute_import, division, print_function

import os

import pkg_resources

from glue.external.qt.QtCore import Qt
from glue.external.qt import QtGui, QtCore
from glue import core
from glue.qt import ui, icons
from glue.utils.qt import (QMessageBoxPatched as QMessageBox, mpl_to_qt4_color,
                           qt4_to_mpl_color, tint_pixmap, GlueItemWidget,
                           set_cursor)

def edit_layer_color(layer):
    """ Interactively edit a layer's color """
    initial = mpl_to_qt4_color(layer.style.color, alpha=layer.style.alpha)
    color = QtGui.QColorDialog.getColor(initial, None, "Change layer color",
                                        options=QtGui.QColorDialog.ShowAlphaChannel)
    if color.isValid():
        layer.style.color = qt4_to_mpl_color(color)
        layer.style.alpha = color.alpha() / 256.


def edit_layer_symbol(layer):
    """ Interactively edit a layer's symbol """
    options = ['o', '^', '*', 's']
    try:
        initial = options.index(layer.style.marker)
    except IndexError:
        initial = 0
    symb, isok = QtGui.QInputDialog.getItem(None, 'Pick a Symbol',
                                            'Pick a Symbol',
                                            options, current=initial)
    if isok and symb != layer.style.marker:
        layer.style.marker = symb


def edit_layer_point_size(layer):
    """ Interactively edit a layer's point size """
    size, isok = QtGui.QInputDialog.getInt(None, 'Point Size', 'Point Size',
                                           value=layer.style.markersize,
                                           min=1, max=1000, step=1)
    if isok and size != layer.style.markersize:
        layer.style.markersize = size


def edit_layer_label(layer):
    """ Interactively edit a layer's label """
    label, isok = QtGui.QInputDialog.getText(None, 'New Label:', 'New Label:',
                                             text=layer.label)
    if isok and str(label) != layer.label:
        layer.label = str(label)


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
    color = mpl_to_qt4_color(layer.style.color)
    pm = tint_pixmap(bm, color)
    pm = pm.scaledToHeight(15, Qt.SmoothTransformation)
    return QtGui.QIcon(pm)


def layer_artist_icon(artist):
    """Create a QtGui.QIcon for a LayerArtist instance"""

    from glue.viewers.image.qt.rgb_edit import RGBEdit
    # TODO: need a test for this

    from glue.viewers.image.layer_artist import ImageLayerArtist

    if not artist.enabled:
        bm = QtGui.QBitmap(icon_path('glue_delete'))
    elif isinstance(artist, ImageLayerArtist):
        bm = QtGui.QBitmap(icon_path('glue_image'))
    else:
        bm = QtGui.QBitmap(icon_path(POINT_ICONS.get(artist.layer.style.marker,
                                                     'glue_circle_point')))
    color = mpl_to_qt4_color(artist.layer.style.color)

    pm = tint_pixmap(bm, color)
    return QtGui.QIcon(pm)


class GlueListWidget(GlueItemWidget, QtGui.QListWidget):
    SUPPORTED_MIME_TYPE = LAYERS_MIME_TYPE


class GlueActionButton(QtGui.QPushButton):

    def set_action(self, action, text=True):
        self._text = text
        self._action = action
        self.clicked.connect(action.trigger)
        action.changed.connect(self._sync_to_action)
        self._sync_to_action()

    def _sync_to_action(self):
        self.setIcon(self._action.icon())
        if self._text:
            self.setText(self._action.text())
        self.setToolTip(self._action.toolTip())
        self.setWhatsThis(self._action.whatsThis())
        self.setEnabled(self._action.isEnabled())


def _custom_widgets():
    # iterate over custom widgets referenced in .ui files
    yield GlueListWidget
    yield GlueActionButton
    yield RGBEdit

    from glue.dialogs.common.qt.component_selector import ComponentSelector
    yield ComponentSelector

    from glue.dialogs.link_editor.qt.link_equation import LinkEquation
    yield LinkEquation


def load_ui(path, parent=None, directory=None):
    """
    Load a UI file, given its name.

    This will first check if `path` exists, and if not it will assume it is the
    name of a ui file to search for in the global glue ui directory.

    Parameters
    ----------
    name : str
      Name of ui file to load (without .ui extension)

    parent : QObject
      Object to use as the parent of this widget

    Returns
    -------
    w : QtGui.QWidget
      The new widget
    """

    if directory is not None:
        full_path = os.path.join(directory, path)
    elif not os.path.exists(path):
        full_path = global_ui_path(path)

    from glue.external.qt import load_ui
    return load_ui(full_path, parent, custom_widgets=_custom_widgets())


def global_ui_path(ui_name):
    """
    Return the absolute path to a .ui file bundled with glue.

    Parameters
    ----------
    ui_name : str
      The name of a ui_file to load (without directory prefix or
      file extensions)

    Returns
    -------
    path : str
      Path of a file
    """
    if not ui_name.endswith('.ui'):
        ui_name = ui_name + '.ui'

    try:
        result = pkg_resources.resource_filename('glue.qt.ui', ui_name)
        return result
    except NotImplementedError:
        # workaround for mac app
        result = os.path.dirname(ui.__file__)
        return os.path.join(result.replace('site-packages.zip', 'glue'),
                            ui_name)


def icon_path(icon_name):
    """Return the absolute path to an icon

    Parameters
    ----------
    icon_name : str
       Name of icon, without extension or directory prefix

    Returns
    -------
    path : str
      Full path to icon
    """
    if not icon_name.endswith('.png'):
        icon_name += '.png'

    try:
        rc = icon_name
        if pkg_resources.resource_exists('glue.qt.icons', rc):
            result = pkg_resources.resource_filename('glue.qt.icons', rc)
            return result
        else:
            raise RuntimeError("Icon does not exist: %s" % icon_name)
    except NotImplementedError:  # workaround for mac app
        result = os.path.dirname(icons.__file__)
        return os.path.join(result.replace('site-packages.zip', 'glue'),
                            icon_name)


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


def action(name, parent, tip='', icon=None, shortcut=None):
    """ Factory for making a new action """
    a = QtGui.QAction(name, parent)
    a.setToolTip(tip)
    if icon:
        a.setIcon(get_icon(icon))
    if shortcut:
        a.setShortcut(shortcut)
    return a


if __name__ == "__main__":

    from glue.qt import get_qapp

    class Foo(object):
        layer_visible = {}
        layer = None

        def update(self):
            print('update', self.layer_visible)

        def redraw(self):
            print('draw')

    app = get_qapp()
    f = Foo()

    rgb = RGBEdit()
    rgb.show()
    app.exec_()

    print(f.layer_visible)
    print(f.contrast_layer)
