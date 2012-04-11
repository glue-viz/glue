from PyQt4.QtGui import QApplication, QMainWindow
import sys

import cloudviz as cv
from cloudviz.qt.cloudviz_toolbar import CloudvizToolbar
from cloudviz.qt.messagewidget import MessageWidget
from cloudviz.qt.scatterwidget import ScatterWidget

def main():

    app = QApplication(sys.argv)
    win = QMainWindow()

    data, data2, s, s2 = cv.example_data.pipe()
    dc = cv.DataCollection([data, data2])
    scatter_client = ScatterWidget(dc)
    message_client = MessageWidget()
    tb = CloudvizToolbar(dc,
                         scatter_client.ui.mplWidget.canvas,
                         frame = scatter_client)

    hub = cv.Hub(data, data2, dc, s, s2, scatter_client, message_client, tb)
    scatter_client.add_layer(data)
    scatter_client.add_layer(data2)

    win.setCentralWidget(scatter_client)
    win.addToolBar(tb)

    win.show()
    message_client.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()