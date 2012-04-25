from PyQt4.QtCore import Qt, SIGNAL
from PyQt4.QtGui import *

from scatterwidget import ScatterWidget
from imagewidget import ImageWidget

class GlueVizLoaderWidget(QWidget):

    @staticmethod
    def wrapper_factory(app, parent=None):
        parent = QWidget(parent)
        child = GlueVizLoaderWidget(app, parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addWidget(child)
        parent.setLayout(layout)
        return parent

    def __init__(self, app, parent):
        super(GlueVizLoaderWidget, self).__init__(parent)

        self.options = [ScatterWidget, ImageWidget]
        self.names = ['Scatter Plot', 'Image Viewer']
        self.data = app._data
        self.hub = app._hub
        self.app = app

        dl = QComboBox()
        map(lambda x: dl.addItem("%s" % x), self.names)
        layout = QVBoxLayout()
        layout.setContentsMargins(2,2,2,2)
        layout.setSpacing(0)
        layout.addWidget(dl)

        ok = QPushButton("OK")
        layout.addWidget(ok)
        self.setLayout(layout)

        ok.pressed.connect(lambda: self.load_viz(dl.currentIndex()))

    def load_viz(self, index):
        parent = self.parent()
        layout = parent.layout()

        print 'creating new widget'
        widget = self.options[index](self.data)
        print 'registering'
        widget.register_to_hub(self.hub)

        print 'removing wizard'
        layout.removeWidget(self)
        self.hide()

        print 'adding widget to layout'
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
