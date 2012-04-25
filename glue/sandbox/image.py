import sys

from PyQt4.QtGui import QApplication, QMainWindow

import glue
from glue.qt.glue_toolbar import GlueToolbar
from glue.qt.messagewidget import MessageWidget
from glue.qt.imagewidget import ImageWidget



def main():

    app = QApplication(sys.argv)
    win = QMainWindow()

    #data, subset = glue.example_data.simple_cube()
    d2, s2 = glue.example_data.simple_image()

    #dc = glue.DataCollection([data, d2, data])
    dc = glue.DataCollection([d2])
    image_client = ImageWidget(dc)
    message_client = MessageWidget()

    hub = glue.Hub(dc, image_client, message_client)
    win.setCentralWidget(image_client)
    win.show()
    message_client.show()
    #image_client.client.set_norm(1, 2000)
    #image_client.client.set_cmap('hot')
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()