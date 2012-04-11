from PyQt4.QtCore import Qt, SIGNAL
from PyQt4.QtGui import *

from scatterwidget import ScatterWidget
from imagewidget import ImageWidget

class GlueVizLoaderWidget(QWidget):

    @staticmethod
    def wrapper_factory(data, parent=None):
        parent = QWidget(parent)
        child = GlueVizLoaderWidget(data, parent)
        layout = QHBoxLayout()
        layout.addWidget(child)
        parent.setLayout(layout)
        return parent

    def __init__(self, data, parent):
        super(GlueVizLoaderWidget, self).__init__(parent)

        self.options = [ScatterWidget, ImageWidget]
        self.data = data

        dl = QComboBox()
        map(lambda x: dl.addItem("%s" % x), self.options)
        layout = QVBoxLayout()
        layout.addWidget(dl)

        ok = QPushButton("OK")
        layout.addWidget(ok)
        self.setLayout(layout)

        ok.pressed.connect(lambda: self.load_viz(dl.currentIndex()))

    def load_viz(self, index):
        parent = self.parent()
        layout = parent.layout()

        widget = self.options[index](self.data, parent)
        layout.removeWidget(self)
        self.hide()
        layout.addWidget(widget)

def test():
    import glue
    import sys

    app = QApplication(sys.argv)

    data, subset = glue.example_data.simple_image()
    dc = glue.DataCollection([data])

    base = QWidget()
    layout = QHBoxLayout()
    base.setLayout(layout)

    layout.addWidget(GlueVizLoaderWidget(dc, base))
    base.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    test()
