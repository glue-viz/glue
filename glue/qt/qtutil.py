from matplotlib.colors import ColorConverter
from PyQt4 import QtGui
from PyQt4.QtCore import Qt
from PyQt4.QtCore import QMimeData
from PyQt4.QtGui import (QColor, QInputDialog, QColorDialog,
                         QListWidget, QTreeWidget, QPushButton, QMessageBox,
                         QTabBar)

from .. import core
from .decorators import set_cursor

def mpl_to_qt4_color(color, alpha=1.0):
    """ Convert a matplotlib color stirng into a PyQT4 QColor object

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
        msg = "\n".join(["Could not load data (wrong load method?)",
                         "File load method: %s" % factory.label,
                         "",
                         "%s" % error,
                         "%s" % traceback.format_exc()])
        ok = QMessageBox.critical(None, "Error loading data", msg,
                                  buttons=buttons, defaultButton=cancel)
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
    if result is not None:
        return [result]
    else:
        return []


class GlueDataDialog(object):

    def __init__(self, parent=None):
        self._fd = QtGui.QFileDialog(parent)
        import glue
        self.filters = [(f[0], self._filter(f[0]))
                        for f in glue.env.data_factories]
        self.setNameFilter()
        self._fd.setFileMode(QtGui.QFileDialog.ExistingFile)

    def factory(self):
        fltr = self._fd.selectedFilter()
        for k, v in self.filters:
            if v == fltr:
                return k

    def setNameFilter(self):
        fltr = ";;".join([flt for fac, flt in self.filters])
        self._fd.setNameFilter(fltr)

    def _filter(self, factory):
        return "%s (%s)" % (factory.label, factory.file_filter)

    def path(self):
        return self._fd.selectedFiles()[0]

    def _get_path_and_factory(self):
        """Show dialog to get a file path and data factory

        :rtype: tuple of (string, func) giving the path and data factory.
                returns (None, None) if user cancel's dialog
        """
        result = self._fd.exec_()
        if result == QtGui.QDialog.Rejected:
            return None, None
        path = self.path()
        factory = self.factory()
        return path, factory

    @set_cursor(Qt.WaitCursor)
    def load_data(self):
        """Highest level method to interactively load a data set.

        :rtype: A constructed data object, or None
        """
        from glue.core.data_factories import data_label
        path, fac = self._get_path_and_factory()
        if path is not None:
            result = fac(path)
            result.label = data_label(path)
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
    dialog = QInputDialog()
    symb, isok = dialog.getItem(None, 'Pick a Symbol',
                                'Pick a Symbol',
                                ['.', 'o', 'v', '>', '<', '^'])
    if isok:
        layer.style.marker = symb


def edit_layer_point_size(layer):
    """ Interactively edit a layer's point size """
    dialog = QInputDialog()
    size, isok = dialog.getInt(None, 'Point Size', 'Point Size',
                               value=layer.style.markersize,
                               min=1, max=1000, step=1)
    if isok:
        layer.style.markersize = size


def edit_layer_label(layer):
    """ Interactively edit a layer's label """
    dialog = QInputDialog()
    label, isok = dialog.getText(None, 'New Label:', 'New Label:',
                                 text=layer.label)
    if isok:
        layer.label = str(label)


def pick_item(items, labels, title="Pick an item", label="Pick an item"):
    """ Prompt the user to choose an item

    :param items: List of items to choose
    :param labels: List of strings to label items
    :param title: Optional widget title
    :param label: Optional prompt

    Returns the selected item, or None
    """
    choice, isok = QInputDialog.getItem(None, title, label, labels)
    if isok:
        return dict(zip(labels, items))[str(choice)]


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
    dialog = QInputDialog()
    result, isok = dialog.getText(None, title, title)
    if isok:
        return str(result)


class PyMimeData(QMimeData):
    """Stores references to live python objects.

    Normal QMimeData instances store all data as QByteArrays. This
    makes it hard to pass around live python objects in drag/drop
    events, since one would have to convert between object references
    and byte sequences.

    The object to store is passed to the constructor, and stored in
    the application/py_instance mime_type.
    """
    MIME_TYPE = 'application/py_instance'

    def __init__(self, instance):
        """
        :param instance: The python object to store
        """
        super(PyMimeData, self).__init__()
        self._instance = instance
        self.setData(self.MIME_TYPE, '1')

    def data(self, mime_type):
        """ Retrieve the data stored at the specified mime_type

        If mime_type is application/py_instance, a python object
        is returned. Otherwise, a QByteArray is returned """
        if str(mime_type) == self.MIME_TYPE:
            return self._instance

        return super(PyMimeData, self).data(mime_type)


class GlueItemView(object):
    """ A partial implementation of QAbstractItemView, with drag events.

    Items can be registered with data via set_data. If the corresponding
    graphical items are dragged, the data will be wrapped in a PyMimeData"""
    def __init__(self, parent=None):
        super(GlueItemView, self).__init__(parent)
        self._mime_data = {}
        self.setDragEnabled(True)

    def mimeTypes(self):
        types = [PyMimeData.MIME_TYPE]
        return types

    def mimeData(self, selected_items):
        assert len(selected_items) == 1
        item = selected_items[0]
        try:
            data = self._mime_data[item]
        except KeyError:
            data = None
        return PyMimeData(data)

    def get_data(self, item):
        return self._mime_data[item]

    def set_data(self, item, data):
        self._mime_data[item] = data

    @property
    def data(self):
        return self._mime_data


class GlueListWidget(GlueItemView, QListWidget):
    pass


class GlueTreeWidget(GlueItemView, QTreeWidget):
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
