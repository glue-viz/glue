from matplotlib.colors import ColorConverter
from matplotlib import cm
import numpy as np
from PyQt4 import QtGui
from PyQt4.QtCore import Qt
from PyQt4.QtCore import QMimeData
from PyQt4.QtGui import (QColor, QInputDialog, QColorDialog,
                         QListWidget, QTreeWidget, QPushButton, QMessageBox,
                         QTabBar, QBitmap, QIcon, QPixmap, QImage,
                         QDialogButtonBox)

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
        from glue.config import data_factory
        self.filters = [(f, self._filter(f))
                        for f in data_factory.members]
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
        return "%s (%s)" % (factory.label, factory.filter)

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
        path = str(self.path())  # cast out of unicode
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
            result = fac.function(path)
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
    options = ['o', '^', '*', 's']
    try:
        initial = options.index(layer.style.marker)
    except IndexError:
        initial = 0
    symb, isok = dialog.getItem(None, 'Pick a Symbol',
                                'Pick a Symbol',
                                options, current=initial)
    if isok and symb != layer.style.marker:
        layer.style.marker = symb


def edit_layer_point_size(layer):
    """ Interactively edit a layer's point size """
    dialog = QInputDialog()
    size, isok = dialog.getInt(None, 'Point Size', 'Point Size',
                               value=layer.style.markersize,
                               min=1, max=1000, step=1)
    if isok and size != layer.style.markersize:
        layer.style.markersize = size


def edit_layer_label(layer):
    """ Interactively edit a layer's label """
    dialog = QInputDialog()
    label, isok = dialog.getText(None, 'New Label:', 'New Label:',
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
                                        labels, current=default)
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


POINT_ICONS = {'o': ':icons/glue_circle_point.png',
               's': ':icons/glue_box_point.png',
               '^': ':icons/glue_triangle_up.png',
               '*': ':icons/glue_star.png',
               '+': ':icons/glue_cross.png'}


def symbol_icon(symbol, color=None):
    bm = QBitmap(POINT_ICONS.get(symbol,
                                 ':icons/glue_circle_point.png'))
    if color is not None:
        return QIcon(tint_pixmap(bm, color))

    return QIcon(bm)


def layer_icon(layer):
    """Create a QIcon for a Data or Subset instance

    :type layer: :class:`~glue.core.data.Data` or
                 :class:`~glue.core.subset.Subset`

    :rtype: QIcon
    """
    bm = QBitmap(POINT_ICONS.get(layer.style.marker,
                                 ':icons/glue_circle_point.png'))
    color = mpl_to_qt4_color(layer.style.color)
    pm = tint_pixmap(bm, color)
    return QIcon(pm)


def layer_artist_icon(artist):
    """Create a QIcon for a LayerArtist instance"""
    from ..clients.layer_artist import ImageLayerArtist
    if isinstance(artist, ImageLayerArtist):
        bm = QBitmap(':icons/glue_image.png')
    else:
        bm = QBitmap(POINT_ICONS.get(artist.layer.style.marker,
                                     ':icons/glue_circle_point.png'))
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


class RGBSelector(QtGui.QDialog):
    def __init__(self, dc, parent=None):
        from .link_equation import ArgumentWidget
        from .component_selector import ComponentSelector

        super(RGBSelector, self).__init__(parent)
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)

        comps = ComponentSelector()
        comps.setup(dc)

        r = ArgumentWidget('r', parent)
        g = ArgumentWidget('g', parent)
        b = ArgumentWidget('b', parent)

        okcancel = QDialogButtonBox(QDialogButtonBox.Ok |
                                    QDialogButtonBox.Cancel)

        layout.addWidget(comps)
        layout.addWidget(r)
        layout.addWidget(g)
        layout.addWidget(b)
        layout.addWidget(okcancel)

        self.r = r
        self.g = g
        self.b = b
        self.component = comps

        okcancel.accepted.connect(self.accept)
        okcancel.rejected.connect(self.reject)

        self.setFocusPolicy(Qt.StrongFocus)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R:
            self.r.component_id = self.component.component
            event.accept()
        elif event.key() == Qt.Key_G:
            self.g.component_id = self.component.component
            event.accept()
        elif event.key() == Qt.Key_B:
            self.b.component_id = self.component.component
            event.accept()
        else:
            super(RGBSelector, self).keyPressEvent(event)


def select_rgb(collect):
    w = RGBSelector(collect)
    result = w.exec_()
    if result == w.Rejected:
        return None
    r = w.r.component_id
    g = w.g.component_id
    b = w.b.component_id
    d = w.component.data

    if r is None or g is None or b is None or d is None:
        return None

    return d, r, g, b
