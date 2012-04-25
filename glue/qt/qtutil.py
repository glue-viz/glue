from matplotlib.colors import ColorConverter
from PyQt4 import QtGui
from PyQt4.QtGui import QColor

import glue

def mpl_to_qt4_color(color):
    """ Convert a matplotlib color stirng into a PyQT4 QColor object

    INPUTS:
    -------
    color: String
       A color specification that matplotlib understands

    RETURNS:
    --------
    A QColor object representing color

    """
    cc = ColorConverter()
    r,g,b = cc.to_rgb(color)
    return QColor(r*255, g*255, b*255)

def qt4_to_mpl_color(color):
    """
    Conver a QColor object into a string that matplotlib understands

    Inputs:
    -------
    color: QColor instance

    OUTPUTS:
    --------
    A hex string describing that color
    """

    hex = color.name()
    return str(hex)


def data_wizard():
    fd = QtGui.QFileDialog()
    if not fd.exec_(): return None
    file = str(fd.selectedFiles()[0])
    extension = file.split('.')[-1].lower()
    label = ' '.join(file.split('.')[:-1])
    label = label.split('/')[-1]
    label = label.split('\\')[-1]

    if extension in ['fits', 'fit', 'fts']:
        result = glue.GriddedData(label=label)
        result.read_data(file)
    else:
        result = glue.data.TabularData(label=label)
        result.read_data(file)
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
            for i,d in enumerate(md.data(f)):
                print i, ("%s" % d)
        self.text_edit.setPlainText("%s" % event.mimeData().text())


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    #print data_wizard()
    dc = DebugClipboard()
    dc.show()
    sys.exit(app.exec_())
