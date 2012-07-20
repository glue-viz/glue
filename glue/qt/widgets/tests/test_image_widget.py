from PyQt4.QtGui import QApplication

from ..image_widget import ImageWidget

from ....tests import example_data
from .... import core

def setup_module(module):
    module.app = QApplication([''])

def teardown_module(module):
    module.app.exit()
    del module.app

class TestImageWidget(object):

    def setup_method(self, method):
        self.hub = core.hub.Hub()
        self.im = example_data.test_image()
        self.cube = example_data.test_cube()
        self.collect = core.data_collection.DataCollection()
        self.widget = ImageWidget(self.collect)
        self.connect_to_hub()
        self.collect.append(self.im)
        self.collect.append(self.cube)

    def connect_to_hub(self):
        self.widget.register_to_hub(self.hub)
        self.collect.register_to_hub(self.hub)

    def teardown_method(self, method):
        self.widget.close()
        del self.widget

    def _test_widget_synced_with_collection(self):
        dc = self.widget.ui.displayDataCombo
        assert dc.count() == len(self.collect)
        for data in self.collect:
            label = data.label
            pos = dc.findText(label)
            assert pos >= 0
            assert dc.itemData(pos) is data

    def test_synced_on_init(self):
        self._test_widget_synced_with_collection()

    def test_multi_add_ignored(self):
        """calling add_data multiple times doesn't corrupt data combo"""
        self.widget.add_data(self.collect[0])
        self.widget.add_data(self.collect[0])
        self._test_widget_synced_with_collection()

    def test_synced_on_remove(self):
        self.collect.remove(self.cube)
        self._test_widget_synced_with_collection()
