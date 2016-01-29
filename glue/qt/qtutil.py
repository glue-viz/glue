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


def data_wizard():
    """ QT Dialog to load a file into a new data object

    Returns:
       A list of new data objects. Returns an empty list if
       selection is canceled.
    """
    def report_error(error, factory):
        import traceback
        retry = QMessageBox.Retry
        cancel = QMessageBox.Cancel
        buttons = retry | cancel
        detail = traceback.format_exc()
        msg = "\n".join(["Could not load data (wrong load method?)",
                         "File load method: %s" % factory.label])
        detail = "\n\n".join(["Error message: %s" % error, detail])
        mb = QMessageBox(QMessageBox.Critical, "Data Load Error", msg)
        mb.setDetailedText(detail)
        mb.setDefaultButton(cancel)
        mb.setStandardButtons(buttons)
        ok = mb.exec_()
        return ok == retry

    while True:
        gdd = GlueDataDialog()
        try:
            result = gdd.load_data()
            break
        except Exception as e:
            decision = report_error(e, gdd.factory())
            if not decision:
                return []
    return result


class GlueDataDialog(object):

    def __init__(self, parent=None):
        self._fd = QtGui.QFileDialog(parent)
        from glue.config import data_factory
        self.filters = [(f, self._filter(f))
                        for f in data_factory.members if not f.deprecated]
        self.setNameFilter()
        self._fd.setFileMode(QtGui.QFileDialog.ExistingFiles)
        try:
            self._fd.setOption(QtGui.QFileDialog.Option.HideNameFilterDetails,
                               True)
        except AttributeError:  # HideNameFilterDetails not present
            pass

    def factory(self):
        fltr = self._fd.selectedNameFilter()
        for k, v in self.filters:
            if v.startswith(fltr):
                return k

    def setNameFilter(self):
        fltr = ";;".join([flt for fac, flt in self.filters])
        self._fd.setNameFilter(fltr)

    def _filter(self, factory):
        return "%s (*)" % factory.label

    def paths(self):
        """
        Return all selected paths, as a list of unicode strings
        """
        return self._fd.selectedFiles()

    def _get_paths_and_factory(self):
        """Show dialog to get a file path and data factory

        :rtype: tuple of (list-of-strings, func)
                giving the path and data factory.
                returns ([], None) if user cancels dialog
        """
        result = self._fd.exec_()
        if result == QtGui.QDialog.Rejected:
            return [], None
        # path = list(map(str, self.paths()))  # cast out of unicode
        path = list(self.paths())
        factory = self.factory()
        return path, factory

    @set_cursor(Qt.WaitCursor)
    def load_data(self):
        """Highest level method to interactively load a data set.

        :rtype: A list of constructed data objects
        """
        from glue.core.data_factories import data_label, load_data
        paths, fac = self._get_paths_and_factory()
        result = []

        for path in paths:
            d = load_data(path, factory=fac.function)
            if not isinstance(d, list):
                d.label = data_label(path)
                d = [d]
            result.extend(d)

        return result


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


class RGBEdit(QtGui.QWidget):

    """A widget to set the contrast for individual layers in an RGB image

    Based off the ds9 RGB Frame widget

    :param artist: A :class:`~glue.viewers.image.layer_artist.RGBArtistLayerArtist`
                   instance to control

    :param parent: Optional widget parent

    This widget sets the state of the artist object, such that contrast
    adjustments from a :class:`~glue.viewers.image.client` affect
    a particular RGB slice
    """
    current_changed = QtCore.Signal(str)
    colors_changed = QtCore.Signal()

    def __init__(self, parent=None, artist=None):
        super(RGBEdit, self).__init__(parent)
        self._artist = artist

        l = QtGui.QGridLayout()

        current = QtGui.QLabel("Contrast")
        visible = QtGui.QLabel("Visible")
        l.addWidget(current, 0, 2, 1, 1)
        l.addWidget(visible, 0, 3, 1, 1)
        l.setColumnStretch(0, 0)
        l.setColumnStretch(1, 10)
        l.setColumnStretch(2, 0)
        l.setColumnStretch(3, 0)

        l.setRowStretch(0, 0)
        l.setRowStretch(1, 0)
        l.setRowStretch(2, 0)
        l.setRowStretch(3, 0)
        l.setRowStretch(4, 10)

        curr_grp = QtGui.QButtonGroup()
        self.current = {}
        self.vis = {}
        self.cid = {}

        for row, color in enumerate(['red', 'green', 'blue'], 1):
            lbl = QtGui.QLabel(color.title())

            cid = ComponentIDCombo()

            curr = QtGui.QRadioButton()
            curr_grp.addButton(curr)

            vis = QtGui.QCheckBox()
            vis.setChecked(True)

            l.addWidget(lbl, row, 0, 1, 1)
            l.addWidget(cid, row, 1, 1, 1)
            l.addWidget(curr, row, 2, 1, 1)
            l.addWidget(vis, row, 3, 1, 1)

            curr.clicked.connect(self.update_current)
            vis.toggled.connect(self.update_visible)
            cid.currentIndexChanged.connect(self.update_layers)

            self.cid[color] = cid
            self.vis[color] = vis
            self.current[color] = curr

        self.setLayout(l)
        self.current['red'].click()

    @property
    def attributes(self):
        """A 3-tuple of the ComponentIDs for each RGB layer"""
        return tuple(self.cid[c].component for c in ['red', 'green', 'blue'])

    @attributes.setter
    def attributes(self, cids):
        for cid, c in zip(cids, ['red', 'green', 'blue']):
            if cid is None:
                continue
            self.cid[c].component = cid

    @property
    def rgb_visible(self):
        """ A 3-tuple of the visibility of each layer, as bools """
        return tuple(self.vis[c].isChecked() for c in ['red', 'green', 'blue'])

    @rgb_visible.setter
    def rgb_visible(self, value):
        for v, c in zip(value, 'red green blue'.split()):
            self.vis[c].setChecked(v)

    @property
    def artist(self):
        return self._artist

    @artist.setter
    def artist(self, value):
        self._artist = value
        for cid in self.cid.values():
            cid.data = value.layer
        self.update_layers()

    def update_layers(self):
        if self.artist is None:
            return

        r = self.cid['red'].component
        g = self.cid['green'].component
        b = self.cid['blue'].component
        changed = self.artist.r is not r or \
            self.artist.g is not g or\
            self.artist.b is not b

        self.artist.r = r
        self.artist.g = g
        self.artist.b = b

        if changed:
            self.colors_changed.emit()

        self.artist.update()
        self.artist.redraw()

    def update_current(self, *args):
        if self.artist is None:
            return

        for c in ['red', 'green', 'blue']:
            if self.current[c].isChecked():
                self.artist.contrast_layer = c
                self.current_changed.emit(c)
                break
        else:
            raise RuntimeError("Could not determine which layer is current")

    def update_visible(self, *args):
        if self.artist is None:
            return

        self.artist.layer_visible['red'] = self.vis['red'].isChecked()
        self.artist.layer_visible['green'] = self.vis['green'].isChecked()
        self.artist.layer_visible['blue'] = self.vis['blue'].isChecked()
        self.artist.update()
        self.artist.redraw()


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


class ComponentIDCombo(QtGui.QComboBox, core.HubListener):

    """ A widget to select among componentIDs in a dataset """

    def __init__(self, data=None, parent=None, visible_only=True):
        QtGui.QComboBox.__init__(self, parent)
        self._data = data
        self._visible_only = visible_only

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if value is None:
            return
        self._data = value
        if value.hub is not None:
            self.register_to_hub(value.hub)
        self.refresh_components()

    @property
    def component(self):
        return self.itemData(self.currentIndex())

    @component.setter
    def component(self, value):
        for i in range(self.count()):
            if self.itemData(i) is value:
                self.setCurrentIndex(i)
                return
        else:
            raise ValueError("Unable to select %s" % value)

    def refresh_components(self):
        if self.data is None:
            return

        self.blockSignals(True)
        old_data = self.itemData(self.currentIndex())

        self.clear()
        if self._visible_only:
            fields = self.data.visible_components
        else:
            fields = self.data.components

        index = 0
        for i, f in enumerate(fields):
            self.addItem(f.label, userData=f)
            if f == old_data:
                index = i

        self.blockSignals(False)
        self.setCurrentIndex(index)

    def register_to_hub(self, hub):
        hub.subscribe(self,
                      core.message.ComponentsChangedMessage,
                      handler=lambda x: self.refresh_components,
                      filter=lambda x: x.data is self._data)


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
