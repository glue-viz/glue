import unittest

from PyQt4.QtGui import QApplication, QMainWindow
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt

import glue
from glue.qt.imagewidget import ImageWidget

class TestImageWidget(unittest.TestCase):

    def setUp(self):
        import sys
        self.app = QApplication(sys.argv)
        self.hub = glue.Hub()
        self.im = glue.example_data.test_image()
        self.cube = glue.example_data.test_cube()
        self.collect = glue.DataCollection()
        self.widget = ImageWidget(self.collect)
        self.collect.append(self.im)
        self.collect.append(self.cube)
        self.connect_to_hub()
        self.widget.show()

    def connect_to_hub(self):

        self.widget.register_to_hub(self.hub)
        self.collect.register_to_hub(self.hub)


    def tearDown(self):
        self.widget.close()
        self.app.exit()
        del self.widget
        del self.app

    def test_set_data_via_method(self):
        self.widget.add_data(self.im)
        self.assertTrue(self.widget.client.display_data == self.im)
