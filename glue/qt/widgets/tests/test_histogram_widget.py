#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
import pytest

from ..histogram_widget import HistogramWidget
from .... import core


def mock_data():
    return core.Data(x=[1, 2, 3], y=[2, 3, 4])


class TestHistogramWidget(object):
    """ Since widget creation is slow, we try to minimize
    by using class-level variables. Unforunately, the test's aren't isolated
    now"""

    @classmethod
    def setup_class(cls):
        cls.reset()

    @classmethod
    def reset(cls):
        cls.data = mock_data()
        cls.collect = core.data_collection.DataCollection([cls.data])
        cls.widget = HistogramWidget(cls.collect)

    @classmethod
    def teardown_class(cls):
        cls.widget.close()

    def set_up_hub(self):
        hub = core.hub.Hub()
        self.collect.register_to_hub(hub)
        self.widget.register_to_hub(hub)
        return hub

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
        print list(self.widget.client._artists)

        comps = set(c for d in self.collect for c in d.visible_components)
        assert self.widget.ui.attributeCombo.count() == len(comps)
        for i in range(self.widget.ui.attributeCombo.count()):
            data = self.widget.ui.attributeCombo.itemData(i)
            assert data in comps

    def test_double_add_ignored(self):
        self.widget.add_data(self.data)
        self.widget.add_data(self.data)

    def test_remove_data(self):
        """ should remove entry fom combo box """
        hub = self.set_up_hub()
        self.widget.add_data(self.data)
        self.collect.remove(self.data)
        assert not self.widget.data_present(self.data)
        self.reset()

    def test_remove_all_data(self):
        self.set_up_hub()
        self.collect.append(core.Data())
        for data in list(self.collect):
            self.collect.remove(data)
            assert not self.widget.data_present(self.data)
        self.reset()

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

    def test_update_xmin(self):
        self.reset()
        self.widget.ui.xmin.setText('-5')
        self.widget._set_limits()
        assert self.widget.client.xlimits[0] == -5

    def test_update_xmax(self):
        self.widget.ui.xmin.setText('15')
        self.widget._set_limits()
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
