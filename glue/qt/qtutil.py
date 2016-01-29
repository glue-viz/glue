"""
Various standalone utility code for
working with Qt
"""

from __future__ import absolute_import, division, print_function

import os

import pkg_resources

from glue.external.qt.QtCore import Qt
from glue.external.qt import QtGui
from glue.qt import ui, icons
from glue.core.qt.mime import LAYERS_MIME_TYPE
from glue.utils.qt import mpl_to_qt4_color, tint_pixmap, GlueItemWidget

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

    from glue.core.qt.mime import GlueMimeListWidget
    yield GlueMimeListWidget
    
    yield GlueActionButton

    from glue.viewers.image.qt.rgb_edit import RGBEdit
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
