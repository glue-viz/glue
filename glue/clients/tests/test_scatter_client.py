import pytest

import numpy as np
import matplotlib.pyplot as plt
from mock import MagicMock

from ...tests import example_data
from ... import core

from ..scatter_client import ScatterClient

# share matplotlib instance, and disable rendering, for speed
FIGURE = plt.figure()
AXES = FIGURE.add_subplot(111)
FIGURE.canvas.draw = lambda: 0
plt.close('all')


class TestScatterClient(object):

    def setup_method(self, method):
        self.data = example_data.test_data()
        self.ids = [self.data[0].find_component_id('a')[0],
                    self.data[0].find_component_id('b')[0],
                    self.data[1].find_component_id('c')[0],
                    self.data[1].find_component_id('d')[0]]
        self.hub = core.hub.Hub()
        self.collect = core.data_collection.DataCollection()
        self.client = ScatterClient(self.collect, axes=AXES)
        self.connect()

    def add_data(self, data=None):
        if data == None:
            data = self.data[0]
        self.collect.append(data)
        self.client.add_data(data)
        return data

    def add_data_and_attributes(self):
        data = self.add_data()
        self.client.set_xdata(self.ids[0])
        self.client.set_ydata(self.ids[1])
        return data

    def is_first_in_front(self, front, back):
        z1 = self.client.get_layer_order(front)
        z2 = self.client.get_layer_order(back)
        return z1 > z2

    def connect(self):
        self.client.register_to_hub(self.hub)
        self.collect.register_to_hub(self.hub)

    def layer_drawn(self, layer):
        return self.client.is_layer_present(layer) and \
            self.client.managers[layer].is_visible() and \
            self.client.managers[layer].is_enabled()

    def layer_data_correct(self, layer, x, y):
        artist = self.client.managers[layer]._artist
        xy = artist.get_offsets()
        if max(abs(xy[:, 0] - x)) > .01:
            return False
        if max(abs(xy[:, 1] - y)) > .01:
            return False
        return True

    def test_empty_on_creation(self):
        for d in self.data:
            assert not self.client.is_layer_present(d)

    def test_add_external_data_raises_exception(self):
        data = core.data.Data()
        with pytest.raises(TypeError) as exc:
            self.client.add_data(data)
        assert exc.value.args[0] == "Layer not in data collection"

    def test_valid_add(self):
        layer = self.add_data()
        assert self.client.is_layer_present(self.data[0])

    def test_axis_labels_sync_with_setters(self):
        layer = self.add_data()
        self.client.set_xdata(self.ids[1])
        assert self.client.ax.get_xlabel() == self.ids[1].label
        self.client.set_ydata(self.ids[0])
        assert self.client.ax.get_ylabel() == self.ids[0].label

    def test_logs(self):
        layer = self.add_data()
        self.client.set_xlog(True)
        assert self.client.ax.get_xscale() == 'log'

        self.client.set_xlog(False)
        assert self.client.ax.get_xscale() == 'linear'

        self.client.set_ylog(True)
        assert self.client.ax.get_yscale() == 'log'

        self.client.set_ylog(False)
        assert self.client.ax.get_yscale() == 'linear'

    def test_flips(self):
        layer = self.add_data()

        self.client.set_xflip(True)
        assert self.client.is_xflip()

        self.client.set_xflip(False)
        assert not self.client.is_xflip()

        self.client.set_yflip(True)
        assert self.client.is_yflip()

        self.client.set_yflip(False)
        assert not self.client.is_yflip()

    def test_double_add(self):
        n0 = len(self.client.ax.collections)
        layer = self.add_data()
        #data and edit_subset present
        assert len(self.client.ax.collections) == 2 + n0
        layer = self.add_data()
        #data and edit_subset still present
        assert len(self.client.ax.collections) == 2 + n0

    def test_data_updates_propagate(self):
        layer = self.add_data_and_attributes()
        assert self.layer_drawn(layer)
        self.client._layer_updated = False
        layer.style.color = 'k'
        assert self.client._layer_updated

    def test_data_removal(self):
        layer = self.add_data()
        subset = layer.new_subset()
        self.collect.remove(layer)
        assert not self.client.is_layer_present(layer)
        assert not self.client.is_layer_present(subset)

    def test_add_subset_while_connected(self):
        layer = self.add_data()
        subset = layer.new_subset()
        assert self.client.is_layer_present(subset)

    def test_subset_removal(self):
        layer = self.add_data()
        subset = layer.new_subset()
        assert self.client.is_layer_present(layer)
        subset.delete()
        assert not self.client.is_layer_present(subset)

    def test_add_subset_to_untracked_data(self):
        subset = self.data[0].new_subset()
        assert not self.client.is_layer_present(subset)

    def test_valid_plot_data(self):
        layer = self.add_data_and_attributes()
        x = layer[self.ids[0]]
        y = layer[self.ids[1]]
        assert self.layer_data_correct(layer, x, y)

    def test_attribute_update_plot_data(self):
        layer = self.add_data_and_attributes()
        x = layer[self.ids[0]]
        y = layer[self.ids[0]]
        self.client.set_ydata(self.ids[0])
        assert self.layer_data_correct(layer, x, y)

    def test_invalid_plot(self):
        layer = self.add_data_and_attributes()
        assert self.layer_drawn(layer)
        c = core.data.ComponentID('bad id')
        self.client.set_xdata(c)
        assert not self.layer_drawn(layer)

    def test_redraw_called_on_invalid_plot(self):
        """ Plot should be updated when given invalid data,
        to sync layers' disabled/invisible states"""
        ctr = MagicMock()
        layer = self.add_data_and_attributes()
        assert self.layer_drawn(layer)
        c = core.data.ComponentID('bad id')
        self.client._redraw = ctr
        ct0 = ctr.call_count
        self.client.set_xdata(c)
        ct1 = ctr.call_count
        ncall = ct1 - ct0
        expected = len(self.client.managers)
        assert ncall == expected

    def test_two_incompatible_data(self):
        d0 = self.add_data(self.data[0])
        d1 = self.add_data(self.data[1])
        self.client.set_xdata(self.ids[0])
        self.client.set_ydata(self.ids[1])
        x = d0[self.ids[0]]
        y = d0[self.ids[1]]
        assert self.layer_drawn(d0)
        assert self.layer_data_correct(d0, x, y)
        assert not self.layer_drawn(d1)

        self.client.set_xdata(self.ids[2])
        self.client.set_ydata(self.ids[3])
        x = d1[self.ids[2]]
        y = d1[self.ids[3]]
        assert self.layer_drawn(d1)
        assert self.layer_data_correct(d1, x, y)
        assert not self.layer_drawn(d0)

    def test_subsets_connect_with_data(self):
        data = self.data[0]
        s1 = data.new_subset()
        s2 = data.new_subset()
        self.collect.append(data)
        self.client.add_data(data)
        assert self.client.is_layer_present(s1)
        assert self.client.is_layer_present(s2)
        assert self.client.is_layer_present(data)

        # should also work with add_layer
        self.collect.remove(data)
        assert data not in self.collect
        assert not self.client.is_layer_present(s1)
        self.collect.append(data)
        self.client.add_layer(data)
        assert self.client.is_layer_present(s1)

    def test_edit_subset_connect_with_data(self):
        data = self.add_data()
        assert self.client.is_layer_present(data.edit_subset)

    def test_edit_subset_removed_with_data(self):
        data = self.add_data()
        self.collect.remove(data)
        assert not self.client.is_layer_present(data.edit_subset)

    def test_apply_roi(self):
        data = self.add_data_and_attributes()
        roi = core.roi.RectangularROI()
        roi.update_limits(.5, .5, 1.5, 1.5)
        x = np.array([1])
        y = np.array([1])
        self.client._apply_roi(roi)
        assert self.layer_data_correct(data.edit_subset, x, y)

    def test_subsets_drawn_over_data(self):
        data = self.add_data_and_attributes()
        subset = data.new_subset()
        assert self.is_first_in_front(subset, data)

    def test_log_sticky(self):
        data = self.add_data_and_attributes()
        assert not self.client.is_xlog()
        assert not self.client.is_ylog()
        self.client.set_xlog(True)
        self.client.set_ylog(True)
        assert self.client.is_xlog()
        assert self.client.is_ylog()
        self.client.set_xdata(data.find_component_id('b')[0])
        self.client.set_ydata(data.find_component_id('b')[0])
        assert self.client.is_xlog()
        assert self.client.is_ylog()

    def test_flip_sticky(self):
        data = self.add_data_and_attributes()
        self.client.set_xflip(True)
        assert self.client.is_xflip()
        self.client.set_xdata(data.find_component_id('b')[0])
        assert self.client.is_xflip()
        self.client.set_xdata(data.find_component_id('a')[0])
        assert self.client.is_xflip()

    def test_visibility_sticky(self):
        data = self.add_data_and_attributes()
        roi = core.roi.RectangularROI()
        roi.update_limits(.5, .5, 1.5, 1.5)
        assert self.client.is_visible(data.edit_subset)
        self.client._apply_roi(roi)
        self.client.set_visible(data.edit_subset, False)
        assert not self.client.is_visible(data.edit_subset)
        self.client._apply_roi(roi)
        assert not self.client.is_visible(data.edit_subset)

    def test_2d_data(self):
        comp = core.data.Component(np.array([[1, 2], [3, 4]]))
        data = core.data.Data()
        cid = data.add_component(comp, '2d')
        self.collect.append(data)
        self.client.add_layer(data)
        self.client.set_xdata(cid)
        self.client.set_ydata(cid)
