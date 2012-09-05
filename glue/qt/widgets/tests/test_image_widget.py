#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
from PyQt4.QtGui import QApplication

from ..image_widget import ImageWidget

from ....tests import example_data
from .... import core


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

    def test_window_title_matches_data(self):
        self.widget.add_data(self.collect[0])
        assert self.widget.windowTitle() == self.collect[0].label

    def test_window_title_updates_on_label_change(self):
        self.connect_to_hub()
        self.widget.add_data(self.collect[0])
        self.collect[0].label = 'Changed'
        assert self.widget.windowTitle() == self.collect[0].label

    def test_data_combo_updates_on_change(self):
        self.connect_to_hub()
        self.widget.add_data(self.collect[0])
        self.collect[0].label = 'changed'
        data_labels = self._data_combo_labels()
        assert self.collect[0].label in data_labels

    def _data_combo_labels(self):
        combo = self.widget.ui.displayDataCombo
        return [combo.itemText(i) for i in range(combo.count())]
