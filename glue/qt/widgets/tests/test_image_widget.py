#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
from PyQt4.QtGui import QApplication

from ..image_widget import ImageWidget

from .... import core

import os
os.environ['GLUE_TESTING'] = 'True'


class TestImageWidget(object):

    def setup_method(self, method):
        self.hub = core.hub.Hub()
        self.im = core.Data(label='im',
                            x=[[1, 2], [3, 4]],
                            y=[[2, 3], [4, 5]])
        self.cube = core.Data(label='cube',
                              x=[[[1, 2], [3, 4]], [[1, 2], [3, 4]]],
                              y=[[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
        self.collect = core.data_collection.DataCollection()
        self.widget = ImageWidget(self.collect)
        self.connect_to_hub()
        self.collect.append(self.im)
        self.collect.append(self.cube)

    def assert_title_correct(self):
        expected = "%s - %s" % (self.widget.client.display_data.label,
                                self.widget.client.display_attribute.label)
        assert self.widget.windowTitle() == expected

    def connect_to_hub(self):
        self.widget.register_to_hub(self.hub)
        self.collect.register_to_hub(self.hub)

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
        self.assert_title_correct()

    def test_window_title_updates_on_label_change(self):
        self.connect_to_hub()
        self.widget.add_data(self.collect[0])
        self.collect[0].label = 'Changed'
        self.assert_title_correct()

    def test_window_title_updates_on_component_change(self):
        self.connect_to_hub()
        self.widget.add_data(self.collect[0])
        self.widget.ui.attributeComboBox.setCurrentIndex(1)
        self.assert_title_correct()

    def test_data_combo_updates_on_change(self):
        self.connect_to_hub()
        self.widget.add_data(self.collect[0])
        self.collect[0].label = 'changed'
        data_labels = self._data_combo_labels()
        assert self.collect[0].label in data_labels

    def _data_combo_labels(self):
        combo = self.widget.ui.displayDataCombo
        return [combo.itemText(i) for i in range(combo.count())]

    def test_data_not_added_on_init(self):
        w = ImageWidget(self.collect)
        assert self.im not in w.client.artists

    def test_selection_switched_on_add(self):
        w = ImageWidget(self.collect)
        assert self.im not in w.client.artists
        w.add_data(self.im)
        assert self.im in w.client.artists
        w.add_data(self.cube)
        assert self.im not in w.client.artists
        assert self.cube in w.client.artists

    def test_component_add_updates_combo(self):
        self.widget.add_data(self.im)
        self.im.add_component(self.im[self.im.components[0]], 'testing')
        combo = self.widget.ui.attributeComboBox
        cids = [combo.itemText(i) for i in range(combo.count())]
        assert 'testing' in cids

    def test_image_correct_on_init_if_first_attribute_hidden(self):
        """Regression test for #127"""
        self.im.components[0]._hidden = True
        self.widget.add_data(self.im)
        combo = self.widget.ui.attributeComboBox
        index = combo.currentIndex()
        assert self.widget.client.display_attribute is combo.itemData(index)
