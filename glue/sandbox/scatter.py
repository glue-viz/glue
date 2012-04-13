from PyQt4.QtGui import QApplication, QMainWindow
import sys

import glue
from glue.qt.glue_toolbar import GlueToolbar
from glue.qt.messagewidget import MessageWidget
from glue.qt.scatterwidget import ScatterWidget
from glue.qt.mouse_mode import RectangleMode

def main():

    app = QApplication(sys.argv)
    win = QMainWindow()

    data, data2, s, s2 = glue.example_data.pipe()
    dc = glue.DataCollection([data, data2])
    scatter_client = ScatterWidget(dc)
    message_client = MessageWidget()
    hub = glue.Hub(data, data2, dc, scatter_client,
                   data.edit_subset, message_client)
    scatter_client.add_layer(data)
    scatter_client.add_layer(data2)
    win.setCentralWidget(scatter_client)
    tb = scatter_client.make_toolbar()
    win.addToolBar(tb)

    win.show()
    #message_client.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()