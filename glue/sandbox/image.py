import sys

from PyQt4.QtGui import QApplication, QMainWindow

import glue
from glue.qt.glue_toolbar import GlueToolbar
from glue.qt.messagewidget import MessageWidget
from glue.qt.imagewidget import ImageWidget



def main():

    app = QApplication(sys.argv)
    win = QMainWindow()

    data, subset = glue.example_data.simple_cube()
    d2, s2 = glue.example_data.simple_image()

    dc = glue.DataCollection([data, d2, data])
    image_client = ImageWidget(dc)
    message_client = MessageWidget()

    tb = image_client.make_toolbar()
    win.addToolBar(tb)
    hub = glue.Hub(data, subset, d2, s2, dc, image_client, message_client)
    win.setCentralWidget(image_client)
    win.show()
    message_client.show()
    dc.active = subset
    #image_client.client.set_norm(1, 2000)
    #image_client.client.set_cmap('hot')
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()