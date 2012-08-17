import glue
from glue.qt.layertreewidget import LayerTreeWidget


def main():
    """ Display a layer tree """
    from PyQt4.QtGui import QApplication, QMainWindow
    import sys

    hub = glue.Hub()
    app = QApplication(sys.argv)
    win = QMainWindow()
    ltw = LayerTreeWidget()
    ltw.register_to_hub(hub)
    ltw.data_collection.register_to_hub(hub)
    win.setCentralWidget(ltw)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
