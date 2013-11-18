"""
Various standalone utility code for
working with Qt
"""

import pkg_resources
from matplotlib.colors import ColorConverter
from matplotlib import cm
import numpy as np
from ..external.qt import QtGui
from ..external.qt.QtCore import Qt
from ..external.qt.QtGui import (QColor, QInputDialog, QColorDialog,
                                 QListWidget, QTreeWidget, QPushButton,
                                 QMessageBox,
                                 QTabBar, QBitmap, QIcon, QPixmap, QImage,
                                 QDialogButtonBox, QWidget,
                                 QVBoxLayout, QHBoxLayout, QLabel,
                                 QRadioButton, QButtonGroup, QCheckBox,
                                 QSizePolicy)

from .decorators import set_cursor
from .mime import PyMimeData, LAYERS_MIME_TYPE
from ..external.qt import is_pyside


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
                         "File load method: %s" % factory.label,
                         "",
                         "%s" % error])
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
                        for f in data_factory.members]
        self.setNameFilter()
        self._fd.setFileMode(QtGui.QFileDialog.ExistingFile)
        self._fd.setOption(QtGui.QFileDialog.Option.HideNameFilterDetails,
                           True)

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

        :rtype: A list of constructed data objects
        """
        from glue.core.data_factories import data_label
        path, fac = self._get_path_and_factory()
        if path is not None:
            result = fac.function(path)
            if isinstance(result, list):
                return result
            # single data object
            result.label = data_label(path)
            return [result]
        return []


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
                                        labels, current=default)
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

    :type layer: :class:`~glue.core.data.Data` or
                 :class:`~glue.core.subset.Subset`

    :rtype: QIcon
    """
    bm = QBitmap(icon_path(POINT_ICONS.get(layer.style.marker,
                                           'circle_point')))
    color = mpl_to_qt4_color(layer.style.color)
    pm = tint_pixmap(bm, color)
    return QIcon(pm)


def layer_artist_icon(artist):
    """Create a QIcon for a LayerArtist instance"""
    from ..clients.layer_artist import ImageLayerArtist
    if isinstance(artist, ImageLayerArtist):
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


class RGBSelector(QtGui.QDialog):

    """Dialog to select ComponentIDs
       to assign to R, G, B color channels"""

    def __init__(self, dc, default=None, parent=None):
        from .link_equation import ArgumentWidget
        from .component_selector import ComponentSelector

        super(RGBSelector, self).__init__(parent)
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)

        comps = ComponentSelector()
        comps.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        comps.setMinimumHeight(400)
        comps.setMinimumWidth(300)
        comps.setup(dc)
        if default is not None:
            comps.data = default

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

        layout.setStretchFactor(comps, 5)
        for w in [r, g, b, okcancel]:
            layout.setStretchFactor(w, 0)
            layout.setStretchFactor(w, 0)
            layout.setStretchFactor(w, 0)

        self.r = r
        self.g = g
        self.b = b
        self.component = comps

        okcancel.accepted.connect(self.accept)
        okcancel.rejected.connect(self.reject)

        self.setFocusPolicy(Qt.StrongFocus)

    @property
    def data(self):
        # the currently-selected data entry
        return self.component.data

    @property
    def red(self):
        return self.r.component_id

    @property
    def green(self):
        return self.g.component_id

    @property
    def blue(self):
        return self.b.component_id

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


def select_rgb(collect, default=None):
    """
    Present an interface to build an RGB image from a data collection,
    return the dataset and components assigned to each color

    :param collect: Data Collection to use
    :type collect: :class:`~glue.core.DataCollection`

    :param default: Default data set to use

    :rtype: tuple of (dataset, red_id, green_id, blue_id), or None
            dataset is the :class:`~glue.core.Data` object to use
            red_id, green_id, blue_id are :class:`~glue.core.ComponentID`s
            Return None if user cancels the action
    """
    w = RGBSelector(collect, default)
    result = w.exec_()
    if result == w.Rejected:
        return None
    r = w.red
    g = w.green
    b = w.blue
    d = w.data

    if r is None or g is None or b is None or d is None:
        return None

    return d, r, g, b


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

    def __init__(self, artist, parent=None):
        super(RGBEdit, self).__init__(parent)
        l = QVBoxLayout()

        widths = [40, 50, 50]

        lbl = QHBoxLayout()
        junk = QLabel("")
        junk.setMinimumWidth(widths[0])
        current = QLabel("Current")
        current.setMinimumWidth(widths[1])
        visible = QLabel("Visible")
        visible.setMinimumWidth(widths[2])
        lbl.addWidget(junk)
        lbl.addWidget(current)
        lbl.addWidget(visible)

        r = QHBoxLayout()
        rl = QLabel("Red")
        rl.setMinimumWidth(widths[0])
        rc = QRadioButton()
        rc.setMinimumWidth(widths[1])
        rv = QCheckBox()
        rc.setMinimumWidth(widths[2])
        rv.setChecked(True)
        r.addWidget(rl)
        r.addWidget(rc)
        r.addWidget(rv)

        g = QHBoxLayout()
        gl = QLabel("Green")
        gl.setMinimumWidth(widths[0])
        gc = QRadioButton()
        gc.setMinimumWidth(widths[2])
        gv = QCheckBox()
        gv.setMinimumWidth(widths[2])
        gv.setChecked(True)
        g.addWidget(gl)
        g.addWidget(gc)
        g.addWidget(gv)

        b = QHBoxLayout()
        bl = QLabel("Blue")
        bl.setMinimumWidth(widths[0])
        bc = QRadioButton()
        bv = QCheckBox()
        bv.setChecked(True)
        bc.setMinimumWidth(widths[1])
        bv.setMinimumWidth(widths[2])

        b.addWidget(bl)
        b.addWidget(bc)
        b.addWidget(bv)

        l.addLayout(lbl)
        l.addLayout(r)
        l.addLayout(g)
        l.addLayout(b)
        self.setLayout(l)
        self.setWindowFlags(Qt.WindowStaysOnTopHint |
                            Qt.X11BypassWindowManagerHint |
                            Qt.Tool)

        bg = QButtonGroup()
        bg.addButton(rc)
        bg.addButton(gc)
        bg.addButton(bc)

        currents = {rc: 'red', bc: 'blue', gc: 'green'}

        rc.clicked.connect(lambda: setattr(artist, 'contrast_layer', 'red'))
        gc.clicked.connect(lambda: setattr(artist, 'contrast_layer', 'green'))
        bc.clicked.connect(lambda: setattr(artist, 'contrast_layer', 'blue'))

        rv.toggled.connect(self.update_visible)
        gv.toggled.connect(self.update_visible)
        bv.toggled.connect(self.update_visible)

        rc.click()

        self.vis = {'red': rv, 'green': gv, 'blue': bv}
        self.artist = artist
        self.current = dict(red=rc, green=gc, blue=bc)

    def update_visible(self):
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
    from .widgets.data_collection_view import DataCollectionView
    yield DataCollectionView

    from .component_selector import ComponentSelector
    yield ComponentSelector

    from .link_equation import LinkEquation
    yield LinkEquation


def _load_ui_pyside(path, parent):
    from PySide.QtUiTools import QUiLoader
    loader = QUiLoader()

    # must register custom widgets referenced in .ui files
    for w in _custom_widgets():
        loader.registerCustomWidget(w)

    widget = loader.load(path, parent)

    return widget


def _load_ui_pyqt4(path, parent):
    from PyQt4.uic import loadUi
    return loadUi(path, parent)


def load_ui(name, parent=None):
    """
    Load a UI file, given it's name

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
    path = ui_path(name)
    if is_pyside():
        return _load_ui_pyside(path, parent)
    return _load_ui_pyqt4(path, parent)


def ui_path(ui_name):
    """Return the absolute path to a .ui file

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
    assert pkg_resources.resource_exists('glue.qt.ui', ui_name)
    result = pkg_resources.resource_filename('glue.qt.ui', ui_name)
    return result


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
    for ext in ['.png']:
        rc = icon_name + ext
        if pkg_resources.resource_exists('glue.qt.icons', rc):
            result = pkg_resources.resource_filename('glue.qt.icons', rc)
            return result
    else:
        raise RuntimeError("Icon does not exist: %s" % icon_name)


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


if __name__ == "__main__":
    from glue.qt import get_qapp

    class Foo(object):
        layer_visible = {}

        def update(self):
            print 'update', self.layer_visible

        def redraw(self):
            print 'draw'

    app = get_qapp()
    f = Foo()

    rgb = RGBEdit(f)
    rgb.show()
    app.exec_()

    print f.layer_visible
    print f.contrast_layer
