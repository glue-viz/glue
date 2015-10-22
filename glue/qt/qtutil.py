"""
Various standalone utility code for
working with Qt
"""

from __future__ import absolute_import, division, print_function

import os

import pkg_resources
from matplotlib.colors import ColorConverter
from matplotlib import cm
import numpy as np

from ..external.axescache import AxesCache
from ..external.qt import QtGui
from ..external.qt.QtCore import (Qt, QThread, QAbstractListModel, QModelIndex)
from ..external.qt.QtGui import (QColor, QInputDialog, QColorDialog,
                                 QListWidget, QTreeWidget, QPushButton,
                                 QTabBar, QBitmap, QIcon, QPixmap, QImage,
                                 QWidget,
                                 QLabel, QGridLayout,
                                 QRadioButton, QButtonGroup, QCheckBox)
from ..utils.qt import QMessageBoxPatched as QMessageBox

from .decorators import set_cursor
from .mime import PyMimeData, LAYERS_MIME_TYPE
from ..external.qt import is_pyside
from ..external.qt.QtCore import Signal
from .. import core
from . import ui, icons

# We import nonpartial here for convenience
from ..utils import nonpartial


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
        return QColor(0, 0, 0, 0)

    cc = ColorConverter()
    r, g, b = cc.to_rgb(color)
    alpha = max(0, min(255, int(256 * alpha)))
    return QColor(r * 255, g * 255, b * 255, alpha)


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
    color = QColorDialog.getColor(initial, None, "Change layer color",
                                  options=QColorDialog.ShowAlphaChannel)
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
    symb, isok = QInputDialog.getItem(None, 'Pick a Symbol',
                                      'Pick a Symbol',
                                      options, current=initial)
    if isok and symb != layer.style.marker:
        layer.style.marker = symb


def edit_layer_point_size(layer):
    """ Interactively edit a layer's point size """
    size, isok = QInputDialog.getInt(None, 'Point Size', 'Point Size',
                                     value=layer.style.markersize,
                                     min=1, max=1000, step=1)
    if isok and size != layer.style.markersize:
        layer.style.markersize = size


def edit_layer_label(layer):
    """ Interactively edit a layer's label """
    label, isok = QInputDialog.getText(None, 'New Label:', 'New Label:',
                                       text=layer.label)
    if isok and str(label) != layer.label:
        layer.label = str(label)


def pick_item(items, labels, title="Pick an item", label="Pick an item",
              default=0):
    """ Prompt the user to choose an item

    :param items: List of items to choose
    :param labels: List of strings to label items
    :param title: Optional widget title
    :param label: Optional prompt

    Returns the selected item, or None
    """
    choice, isok = QInputDialog.getItem(None, title, label,
                                        labels, current=default,
                                        editable=False)
    if isok:
        index = labels.index(str(choice))
        return items[index]


def pick_class(classes, **kwargs):
    """Prompt the user to pick from a list of classes using QT

    :param classes: list of class objects
    :param title: string of the prompt

    Returns:
       The class that was selected, or None
    """
    def _label(c):
        try:
            return c.LABEL
        except AttributeError:
            return c.__name__

    choices = [_label(c) for c in classes]
    return pick_item(classes, choices, **kwargs)


def get_text(title='Enter a label'):
    """Prompt the user to enter text using QT

    :param title: Name of the prompt

    *Returns*
       The text the user typed, or None
    """
    result, isok = QInputDialog.getText(None, title, title)
    if isok:
        return str(result)


class GlueItemWidget(object):

    """ A mixin for QListWidget/GlueTreeWidget subclasses, that
    provides drag+drop funtionality.
    """
    # Implementation detail: QXXWidgetItems are unhashable in PySide,
    # and cannot be used as dictionary keys. we hash on IDs instead

    def __init__(self, parent=None):
        super(GlueItemWidget, self).__init__(parent)
        self._mime_data = {}
        self.setDragEnabled(True)

    def mimeTypes(self):
        """Return the list of MIME Types supported for this object"""
        types = [LAYERS_MIME_TYPE]
        return types

    def mimeData(self, selected_items):
        """Return a list of MIME data associated with the each selected item

        :param selected_items: List of QListWidgetItems or QTreeWidgetItems
        :rtype: List of MIME objects
        """
        try:
            data = [self.get_data(i) for i in selected_items]
        except KeyError:
            data = None
        result = PyMimeData(data, **{LAYERS_MIME_TYPE: data})

        # apparent bug in pyside garbage collects custom mime
        # data, and crashes. Save result here to avoid
        self._mime = result

        return result

    def get_data(self, item):
        """Convenience method to fetch the data associated with a
        QxxWidgetItem"""
        # return item.data(Qt.UserRole)
        return self._mime_data[id(item)]

    def set_data(self, item, data):
        """Convenience method to set data associated with a QxxWidgetItem"""
        #item.setData(Qt.UserRole, data)
        self._mime_data[id(item)] = data

    def drop_data(self, item):
        self._mime_data.pop(id(item))

    @property
    def data(self):
        return self._mime_data


POINT_ICONS = {'o': 'glue_circle_point',
               's': 'glue_box_point',
               '^': 'glue_triangle_up',
               '*': 'glue_star',
               '+': 'glue_cross'}


def symbol_icon(symbol, color=None):
    bm = QBitmap(icon_path(POINT_ICONS.get(symbol, 'glue_circle')))

    if color is not None:
        return QIcon(tint_pixmap(bm, color))

    return QIcon(bm)


def layer_icon(layer):
    """Create a QIcon for a Data or Subset instance

    :type layer: :class:`~glue.core.data.Data`,
                 :class:`~glue.core.subset.Subset`,
                 or object with a .style attribute

    :rtype: QIcon
    """
    icon = POINT_ICONS.get(layer.style.marker, 'circle_point')
    bm = QBitmap(icon_path(icon))
    color = mpl_to_qt4_color(layer.style.color)
    pm = tint_pixmap(bm, color)
    pm = pm.scaledToHeight(15, Qt.SmoothTransformation)
    return QIcon(pm)


def layer_artist_icon(artist):
    """Create a QIcon for a LayerArtist instance"""
    from ..clients.layer_artist import ImageLayerArtist

    if not artist.enabled:
        bm = QBitmap(icon_path('glue_delete'))
    elif isinstance(artist, ImageLayerArtist):
        bm = QBitmap(icon_path('glue_image'))
    else:
        bm = QBitmap(icon_path(POINT_ICONS.get(artist.layer.style.marker,
                                               'glue_circle_point')))
    color = mpl_to_qt4_color(artist.layer.style.color)

    pm = tint_pixmap(bm, color)
    return QIcon(pm)


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
    image.setColor(0, QColor(0, 0, 0, 0).rgba())

    result = QPixmap.fromImage(image)
    return result


class GlueListWidget(GlueItemWidget, QListWidget):
    pass


class GlueTreeWidget(GlueItemWidget, QTreeWidget):
    pass


class GlueActionButton(QPushButton):

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


class GlueTabBar(QTabBar):

    def __init__(self, *args, **kwargs):
        super(GlueTabBar, self).__init__(*args, **kwargs)

    def rename_tab(self, index=None):
        """ Prompt user to rename a tab
        :param index: integer. Index of tab to edit. Defaults to current index
        """
        index = index or self.currentIndex()
        label = get_text("New Tab Label")
        if not label:
            return
        self.setTabText(index, label)

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        index = self.tabAt(event.pos())
        if index >= 0:
            self.rename_tab(index)


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
    rgbas = [QColor(int(r * 255), int(g * 255),
                    int(b * 255), int(a * 255)).rgba() for r, g, b, a in rgbas]
    im = QImage(steps, 1, QImage.Format_Indexed8)
    im.setColorTable(rgbas)
    for i in range(steps):
        im.setPixel(i, 0, i)
    im = im.scaled(100, 100)
    pm = QPixmap.fromImage(im)
    return pm


def pretty_number(numbers):
    """Convert a list of numbers into a nice list of strings

    :param numbers: Numbers to convert
    :type numbers: List or other iterable of numbers

    :rtype: A list of strings
    """
    try:
        return [pretty_number(n) for n in numbers]
    except TypeError:
        pass

    n = numbers
    if n == 0:
        result = '0'
    elif (abs(n) < 1e-3) or (abs(n) > 1e3):
        result = "%0.3e" % n
    elif abs(int(n) - n) < 1e-3 and int(n) != 0:
        result = "%i" % n
    else:
        result = "%0.3f" % n
    if result.find('.') != -1:
        result = result.rstrip('0')

    return result


class RGBEdit(QWidget):

    """A widget to set the contrast for individual layers in an RGB image

    Based off the ds9 RGB Frame widget

    :param artist: A :class:`~glue.clients.layer_artists.RGBLayerArtist`
                   instance to control

    :param parent: Optional widget parent

    This widget sets the state of the artist object, such that contrast
    adjustments from a :class:`~glue.clients.image_client` affect
    a particular RGB slice
    """
    current_changed = Signal(str)
    colors_changed = Signal()

    def __init__(self, parent=None, artist=None):
        super(RGBEdit, self).__init__(parent)
        self._artist = artist

        l = QGridLayout()

        current = QLabel("Contrast")
        visible = QLabel("Visible")
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

        curr_grp = QButtonGroup()
        self.current = {}
        self.vis = {}
        self.cid = {}

        for row, color in enumerate(['red', 'green', 'blue'], 1):
            lbl = QLabel(color.title())

            cid = ComponentIDCombo()

            curr = QRadioButton()
            curr_grp.addButton(curr)

            vis = QCheckBox()
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


class GlueComboBox(QtGui.QComboBox):

    """ Modification of QComboBox, that sidesteps PySide
    sefgaults when storing some python objects as user data
    """

    def __init__(self, parent=None):
        super(GlueComboBox, self).__init__(parent)
        self._data = []

    def addItem(self, text, userData=None):
        # set before super, since super may trigger signals
        self._data.append(userData)
        super(GlueComboBox, self).addItem(text)

    def addItems(self, items):
        self._data.extend(None for _ in items)
        super(GlueComboBox, self).addItems(items)

    def itemData(self, index, role=Qt.UserRole):
        assert len(self._data) == self.count()
        if role != Qt.UserRole:
            return super(GlueComboBox, self).itemData(index, role)
        return self._data[index]

    def setItemData(self, index, value, role=Qt.UserRole):
        if role != Qt.UserRole:
            return super(GlueComboBox, self).setItemData(index, value, role)
        self._data[index] = value

    def clear(self):
        self._data = []
        return super(GlueComboBox, self).clear()

    def insertItem(self, *args):
        raise NotImplementedError()

    def insertItems(self, *args):
        raise NotImplementedError()

    def insertSeparator(self, index):
        raise NotImplementedError()

    def removeItem(self, index):
        self._data.pop(index)
        return super(GlueComboBox, self).removeItem(index)


def _custom_widgets():
    # iterate over custom widgets referenced in .ui files
    yield GlueListWidget
    yield GlueComboBox
    yield GlueActionButton
    yield RGBEdit

    from .component_selector import ComponentSelector
    yield ComponentSelector

    from .link_equation import LinkEquation
    yield LinkEquation


def load_ui(path, parent=None):
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
    w : QWidget
      The new widget
    """

    if not os.path.exists(path):
        path = global_ui_path(path)

    from ..external.qt import load_ui
    return load_ui(path, parent, custom_widgets=_custom_widgets())


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
    Build a QIcon from an image name

    Parameters
    ----------
    icon_name : str
      Name of image file. Assumed to be a png file in glue/qt/icons
      Do not include the extension

    Returns
    -------
    A QIcon object
    """
    return QIcon(icon_path(icon_name))


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


def cache_axes(axes, toolbar):
    """ Setup an caching for an axes object

    After this, cached renders will be used to quickly
    re-render an axes during window resizing or
    interactive pan/zooming.

    :param axes: The matplotlib Axes object to cache
    :param toolbar: The GlueToolbar managing the axes' canvas

    :rtype: The AxesCache instance
    """
    canvas = axes.figure.canvas
    cache = AxesCache(axes)
    canvas.resize_begin.connect(cache.enable)
    canvas.resize_end.connect(cache.disable)
    toolbar.pan_begin.connect(cache.enable)
    toolbar.pan_end.connect(cache.disable)
    return cache


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


class Worker(QThread):
    result = Signal(object)
    error = Signal(object)

    def __init__(self, func, *args, **kwargs):
        """
        Execute a function call on a different QThread

        :param func: The function object to call
        :param args: arguments to pass to the function
        :param kwargs: kwargs to pass to the function
        """
        super(Worker, self).__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """
        Invoke the function
        Upon successful completion, the result signal will be fired
        with the output of the function
        If an exception occurs, the error signal will be fired with
        the result form sys.exc_infno()
        """
        try:
            result = self.func(*self.args, **self.kwargs)
            self.result.emit(result)
        except:
            import sys
            self.error.emit(sys.exc_info())


def update_combobox(combo, labeldata):
    """
    Redefine the items in a combobox

    Parameters
    ----------
    widget : QComboBox
       The widget to update
    labeldata : sequence if N (label, data) tuples
       The combobox will contain N items with the appropriate
       labels, and data set as the userData

    Returns
    -------
    combo : QComboBox
        The updated input

    Notes
    -----
    If the current userData in the combo box matches
    any of labeldata, that selection will be retained.
    Otherwise, the first item will be selected.

    Signals are disabled while the combo box is updated

    combo is modified inplace
    """
    combo.blockSignals(True)
    idx = combo.currentIndex()
    if idx > 0:
        current = combo.itemData(idx)
    else:
        current = None

    combo.clear()
    index = 0
    for i, (label, data) in enumerate(labeldata):
        combo.addItem(label, userData=data)
        if data is current:
            index = i
    combo.blockSignals(False)
    combo.setCurrentIndex(index)

    # We need to force emit this, otherwise if the index happens to be the
    # same as before, even if the data is different, callbacks won't be
    # called.
    if idx == index:
        combo.currentIndexChanged.emit(index)

    return combo


class PythonListModel(QAbstractListModel):

    """
    A Qt Model that wraps a python list, and exposes a list-like interface

    This can be connected directly to multiple QListViews, which will
    stay in sync with the state of the container.
    """

    def __init__(self, items, parent=None):
        """
        Create a new model

        Parameters
        ----------
        items : list
            The initial list to wrap
        parent : QObject
            The model parent
        """
        super(PythonListModel, self).__init__(parent)
        self.items = items

    def rowCount(self, parent=None):
        """Number of rows"""
        return len(self.items)

    def headerData(self, section, orientation, role):
        """Column labels"""
        if role != Qt.DisplayRole:
            return None
        return "%i" % section

    def row_label(self, row):
        """ The textual label for the row"""
        return str(self.items[row])

    def data(self, index, role):
        """Retrieve data at each index"""
        if not index.isValid():
            return None
        if role == Qt.DisplayRole or role == Qt.EditRole:
            return self.row_label(index.row())
        if role == Qt.UserRole:
            return self.items[index.row()]

    def setData(self, index, value, role):
        """
        Update the data in-place

        Parameters
        ----------
        index : QModelIndex
            The location of the change
        value : object
            The new value
        role : QEditRole
            Which aspect of the model to update
        """
        if not index.isValid():
            return False

        if role == Qt.UserRole:
            row = index.row()
            self.items[row] = value
            self.dataChanged.emit(index, index)
            return True

        return super(PythonListModel, self).setDdata(index, value, role)

    def removeRow(self, row, parent=None):
        """
        Remove a row from the table

        Parameters
        ----------
        row : int
            Row to remove

        Returns
        -------
        successful : bool
        """
        if row < 0 or row >= len(self.items):
            return False

        self.beginRemoveRows(QModelIndex(), row, row)
        self._remove_row(row)
        self.endRemoveRows()
        return True

    def pop(self, row=None):
        """
        Remove and return an item (default last item)

        Parameters
        ----------
        row : int (optional)
            Which row to remove. Default=last

        Returns
        --------
        popped : object
        """
        if row is None:
            row = len(self) - 1
        result = self[row]
        self.removeRow(row)
        return result

    def _remove_row(self, row):
        # actually remove data. Subclasses can override this as needed
        self.items.pop(row)

    def __getitem__(self, row):
        return self.items[row]

    def __setitem__(self, row, value):
        index = self.index(row)
        self.setData(index, value, role=Qt.UserRole)

    def __len__(self):
        return len(self.items)

    def insert(self, row, value):
        self.beginInsertRows(QModelIndex(), row, row)
        self.items.insert(row, value)
        self.endInsertRows()
        self.rowsInserted.emit(self.index(row), row, row)

    def append(self, value):
        row = len(self)
        self.insert(row, value)

    def extend(self, values):
        for v in values:
            self.append(v)

    def set_list(self, values):
        """
        Set the model to a new list
        """
        self.beginResetModel()
        self.items = values
        self.endResetModel()
