from matplotlib.colors import ColorConverter
from PyQt4 import QtGui
from PyQt4.QtGui import QColor, QInputDialog, QColorDialog

import glue


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
        result = glue.GriddedData(label=label)
        result.read_data(file_name)
    else:
        result = glue.data.TabularData(label=label)
        result.read_data(file_name)
    return result


class DebugClipboard(QtGui.QWidget):
    """ A simple class to that displays
    any drop event with text data """
    def __init__(self, parent=None):
        super(DebugClipboard, self).__init__(parent)
        self.text_edit = QtGui.QTextEdit(self)
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.text_edit)
        self.setLayout(layout)
        self.text_edit.setAcceptDrops(False)
        self.setAcceptDrops(True)

    def get_text(self):
        return str(self.text_edit.toPlainText())

    def dragEnterEvent(self, event):
        event.accept()
        return
        if event.mimeData().hasText():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        md = event.mimeData()
        print md.hasColor(), md.hasHtml(), md.hasImage()
        print md.hasText(), md.hasUrls()
        print md.text()
        print md.urls()
        for f in md.formats():
            print f
            print type(md.data(f))
            for i, d in enumerate(md.data(f)):
                print i, ("%s" % d)
        self.text_edit.setPlainText("%s" % event.mimeData().text())


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
