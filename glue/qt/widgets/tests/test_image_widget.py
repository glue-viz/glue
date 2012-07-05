import sys

from PyQt4.QtGui import QApplication

from ..image_widget import ImageWidget

from ....tests import example_data
from .... import core


class TestImageWidget(object):

    def setup_method(self, method):

        self.app = QApplication(sys.argv)
        self.hub = core.hub.Hub()
        self.im = example_data.test_image()
        self.cube = example_data.test_cube()
        self.collect = core.data_collection.DataCollection()
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
        assert self.widget.client.display_data == self.im
