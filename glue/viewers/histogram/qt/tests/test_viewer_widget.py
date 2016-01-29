# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import os

import pytest

from glue import core
from glue.core.tests.util import simple_session

from ..viewer_widget import HistogramWidget, _hash


def mock_data():
    return core.Data(label='d1', x=[1, 2, 3], y=[2, 3, 4])

os.environ['GLUE_TESTING'] = 'True'


class TestHistogramWidget(object):

    def setup_method(self, method):
        self.data = mock_data()
        self.session = simple_session()
        self.collect = self.session.data_collection
        self.hub = self.session.hub
        self.collect.append(self.data)
        self.widget = HistogramWidget(self.session)

    def teardown_method(self, method):
        self.widget.close()

    def set_up_hub(self):
        self.collect.register_to_hub(self.hub)
        self.widget.register_to_hub(self.hub)
        return self.hub

    def assert_component_integrity(self, dc=None, widget=None):
        dc = dc or self.collect
        widget = widget or self.widget
        combo = widget.ui.attributeCombo
        row = 0
        for data in dc:
            if data not in widget._layer_artist_container:
                continue
            assert combo.itemText(row) == data.label
            assert combo.itemData(row) == _hash(data)
            row += 2  # next row is separator
            for c in data.visible_components:
                assert combo.itemText(row) == c.label
                assert combo.itemData(row) == _hash(c)
                row += 1

    def test_attribute_set_with_combo(self):
        self.widget.ui.attributeCombo.setCurrentIndex(1)
        obj = self.widget.ui.attributeCombo.itemData(1)
        assert self.widget.client.component is obj

        obj = self.widget.ui.attributeCombo.itemData(0)
        self.widget.ui.attributeCombo.setCurrentIndex(0)
        assert self.widget.client.component is obj

    def test_attributes_populated_after_first_data_add(self):
        d2 = self.data
        self.collect.append(d2)
        self.widget.add_data(d2)
        assert self.widget.client.layer_present(d2)
        print(list(self.widget.client._artists))

        self.assert_component_integrity()

    def test_double_add_ignored(self):
        self.widget.add_data(self.data)
        self.widget.add_data(self.data)

    def test_remove_data(self):
        """ should remove entry fom combo box """
        hub = self.set_up_hub()
        self.widget.add_data(self.data)
        self.collect.remove(self.data)
        assert not self.widget.data_present(self.data)

    def test_remove_all_data(self):
        self.set_up_hub()
        self.collect.append(core.Data())
        for data in list(self.collect):
            self.collect.remove(data)
            assert not self.widget.data_present(self.data)

    @pytest.mark.parametrize(('box', 'prop'),
                             [('normalized_box', 'normed'),
                              ('autoscale_box', 'autoscale'),
                              ('cumulative_box', 'cumulative'),
                              ('xlog_box', 'xlog'),
                              ('ylog_box', 'ylog')])
    def test_check_box_syncs_to_property(self, box, prop):
        box = getattr(self.widget.ui, box)
        box.toggle()
        assert getattr(self.widget.client, prop) == box.isChecked()
        box.toggle()
        assert getattr(self.widget.client, prop) == box.isChecked()

    def test_nbin_change(self):
        self.widget.ui.binSpinBox.setValue(7.0)
        assert self.widget.client.nbins == 7

    def test_update_xmin_xmax(self):

        self.widget.ui.xmin.setText('-5')
        self.widget.ui.xmin.editingFinished.emit()
        assert self.widget.client.xlimits[0] == -5

        self.widget.ui.xmax.setText('15')
        self.widget.ui.xmax.editingFinished.emit()
        assert self.widget.client.xlimits[1] == 15

    def test_update_component_updates_title(self):
        self.widget.add_data(self.data)
        for comp in self.data.visible_components:
            self.widget.component = comp
            assert self.widget.windowTitle() == str(comp)

    def test_update_attributes_preserves_current_component(self):
        self.widget.add_data(self.data)
        self.widget.component = self.data.visible_components[1]
        self.widget._update_attributes()
        assert self.widget.component is self.data.visible_components[1]

    def test_invalid_component_set(self):
        with pytest.raises(IndexError) as exc:
            self.widget.component = None
        assert exc.value.args[0] == "Component not present: None"

    def test_combo_updates_with_component_add(self):
        hub = self.set_up_hub()
        self.widget.add_data(self.data)
        self.data.add_component(self.data[self.data.components[0]], 'testing')
        self.assert_component_integrity()

    def test_nonnumeric_first_component(self):
        # regression test for #208. Shouldn't complain if
        # first component is non-numerical
        data = core.Data()
        data.add_component(['a', 'b', 'c'], label='c1')
        data.add_component([1, 2, 3], label='c2')
        self.collect.append(data)
        self.widget.add_data(data)
