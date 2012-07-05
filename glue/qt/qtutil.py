from matplotlib.colors import ColorConverter
from PyQt4 import QtGui
from PyQt4.QtCore import QMimeData, QStringList
from PyQt4.QtGui import QColor, QInputDialog, QColorDialog, QListWidget

from .. import core


def mpl_to_qt4_color(color):
    """ Convert a matplotlib color stirng into a PyQT4 QColor object

    Parameters
    ----------
    color: String
       A color specification that matplotlib understands

    Returns
    -------
    A QColor object representing color

    """
    cc = ColorConverter()
    r, g, b = cc.to_rgb(color)
    return QColor(r * 255, g * 255, b * 255)


def qt4_to_mpl_color(color):
    """
    Conver a QColor object into a string that matplotlib understands

    Parameters
    ----------
    color: QColor instance

    Returns
    -------
    A hex string describing that color
    """
    hexid = color.name()
    return str(hexid)


def data_wizard():
    """ QT Dialog to load a file into a new data object

    Returns
    -------
    A new data object, or None if the process is cancelled
    """
    fd = QtGui.QFileDialog()
    if not fd.exec_():
        return None
    file_name = str(fd.selectedFiles()[0])
    extension = file_name.split('.')[-1].lower()
    label = ' '.join(file_name.split('.')[:-1])
    label = label.split('/')[-1]
    label = label.split('\\')[-1]

    if extension in ['fits', 'fit', 'fts']:
        result = core.data.GriddedData(label=label)
        result.read_data(file_name)
    else:
        result = core.data.TabularData(label=label)
        result.read_data(file_name)
    return result


def edit_layer_color(layer):
    """ Interactively edit a layer's color """
    dialog = QColorDialog()
    initial = mpl_to_qt4_color(layer.style.color)
    color = dialog.getColor(initial=initial)
    if color.isValid():
        layer.style.color = qt4_to_mpl_color(color)


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
    label, isok = dialog.getText(None, 'New Label:', 'New Label:')
    if isok:
        layer.style.label = str(label)

def pick_class(classes, title="Pick an item"):
    """Prompt the user to pick from a list of classes using QT

    Parameters
    ----------
    classes : list of class objects
    title : string of the prompt

    Outputs
    -------
    The class that was selected, or None
    """

    choices = [c.__name__ for c in classes]

    dialog = QInputDialog()
    choice, isok = dialog.getItem(None, title, title, choices)
    if isok:
        return dict(zip(choices, classes))[str(choice)]

def get_text(title='Enter a label'):
    """Prompt the user to enter text using QT

    Parameters
    ----------
    title : Name of the prompt

    Returns
    -------
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


class GlueListWidget(QListWidget):
    def __init__(self, parent=None):
        super(GlueListWidget, self).__init__(parent)
        self._data = {}

    def mimeTypes(self):
        types = QStringList()
        types.append(PyMimeData.MIME_TYPE)
        return types

    def mimeData(self, selected_items):
        assert len(selected_items) == 1
        item = selected_items[0]
        data = self._data[item]
        return PyMimeData(data)

    @property
    def data(self):
        return self._data
