import unittest
from time import sleep

import numpy as np

import glue
from glue.scatter_client import ScatterClient

class TestScatterClient(unittest.TestCase):

    def setUp(self):
        self.data = glue.example_data.test_data()
        self.ids = self.data[0].component_ids() + self.data[1].component_ids()
        self.hub = glue.Hub()
        self.collect = glue.DataCollection()
        self.client = ScatterClient(self.collect)
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
        z1 = self.client.layers[front]['artist'].get_zorder()
        z2 = self.client.layers[back]['artist'].get_zorder()
        return z1 > z2

    def connect(self):
        self.client.register_to_hub(self.hub)
        self.collect.register_to_hub(self.hub)

    def layer_drawn(self, layer):
        if not self.client.layers[layer]['artist'].get_visible():
            return False
        return True

    def layer_data_correct(self, layer, x, y):
        artist = self.client.layers[layer]['artist']
        xy = artist.get_offsets()
        if max(abs(xy[:,0] - x)) > .01:
            return False
        if max(abs(xy[:,1] - y)) > .01:
            return False
        return True

    def test_empty_on_creation(self):
        for d in self.data:
            self.assertFalse(self.client.is_layer_present(d))

    def test_add_external_data_raises_exception(self):
        data = glue.Data()
        self.assertRaises(TypeError, self.client.add_data, data)

    def test_valid_add(self):
        layer = self.add_data()
        self.assertTrue(self.client.is_layer_present(self.data[0]))

    def test_axis_labels_sync_with_setters(self):
        layer = self.add_data()
        self.client.set_xdata(self.ids[1])
        self.assertEquals(self.client.ax.get_xlabel(), self.ids[1].label)
        self.client.set_ydata(self.ids[0])
        self.assertEquals(self.client.ax.get_ylabel(), self.ids[0].label)

    def test_logs(self):
        layer = self.add_data()
        self.client.set_xlog(True)
        self.assertEquals(self.client.ax.get_xscale(), 'log')

        self.client.set_xlog(False)
        self.assertEquals(self.client.ax.get_xscale(), 'linear')

        self.client.set_ylog(True)
        self.assertEquals(self.client.ax.get_yscale(), 'log')

        self.client.set_ylog(False)
        self.assertEquals(self.client.ax.get_yscale(), 'linear')

    def test_flips(self):
        layer = self.add_data()

        self.client.set_xflip(True)
        self.assertTrue(self.client.is_xflip())

        self.client.set_xflip(False)
        self.assertFalse(self.client.is_xflip())

        self.client.set_yflip(True)
        self.assertTrue(self.client.is_yflip())

        self.client.set_yflip(False)
        self.assertFalse(self.client.is_yflip())

    def test_double_add(self):
        self.assertEquals(len(self.client.ax.collections), 0)
        layer = self.add_data()
        #data and edit_subset present
        self.assertEquals(len(self.client.ax.collections), 2)
        layer = self.add_data()
        #data and edit_subset still present
        self.assertEquals(len(self.client.ax.collections), 2)


    def test_data_updates_propagate(self):
        layer = self.add_data_and_attributes()
        self.assertTrue(self.layer_drawn(layer))
        self.client._layer_updated = False
        layer.style.color = 'k'
        self.assertTrue(self.client._layer_updated)

    def test_data_removal(self):
        layer = self.add_data()
        subset = layer.new_subset()
        self.collect.remove(layer)
        self.assertFalse(self.client.is_layer_present(layer))
        self.assertFalse(self.client.is_layer_present(subset))

    def test_add_subset_while_connected(self):
        layer = self.add_data()
        subset = layer.new_subset()
        self.assertTrue(self.client.is_layer_present(subset))

    def test_subset_removal(self):
        layer = self.add_data()
        subset = layer.new_subset()
        self.assertTrue(self.client.is_layer_present(layer))
        subset.unregister()
        self.assertFalse(self.client.is_layer_present(subset))

    def test_add_subset_to_untracked_data(self):
        subset = self.data[0].new_subset()
        self.assertFalse(self.client.is_layer_present(subset))

    def test_valid_plot_data(self):
        layer = self.add_data_and_attributes()
        x = layer[self.ids[0]]
        y = layer[self.ids[1]]
        self.assertTrue(self.layer_data_correct(layer, x, y))

    def test_attribute_update_plot_data(self):
        layer = self.add_data_and_attributes()
        x = layer[self.ids[0]]
        y = layer[self.ids[0]]
        self.client.set_ydata(self.ids[0])
        self.assertTrue(self.layer_data_correct(layer, x, y))

    def test_invalid_plot(self):
        layer = self.add_data_and_attributes()
        self.assertTrue(self.layer_drawn(layer))
        self.client.set_xdata('attribute_does_not_exist')
        self.assertFalse(self.layer_drawn(layer))

    def test_two_incompatible_data(self):
        d0 = self.add_data(self.data[0])
        d1 = self.add_data(self.data[1])
        self.client.set_xdata(self.ids[0])
        self.client.set_ydata(self.ids[1])
        x = d0[self.ids[0]]
        y = d0[self.ids[1]]
        self.assertTrue(self.layer_drawn(d0))
        self.assertTrue(self.layer_data_correct(d0, x, y))
        self.assertFalse(self.layer_drawn(d1))

        self.client.set_xdata(self.ids[2])
        self.client.set_ydata(self.ids[3])
        x = d1[self.ids[2]]
        y = d1[self.ids[3]]
        self.assertTrue(self.layer_drawn(d1))
        self.assertTrue(self.layer_data_correct(d1, x, y))
        self.assertFalse(self.layer_drawn(d0))

    def test_subsets_connect_with_data(self):
        data = self.data[0]
        s1 = data.new_subset()
        s2 = data.new_subset()
        self.collect.append(data)
        self.client.add_data(data)
        self.assertTrue(self.client.is_layer_present(s1))
        self.assertTrue(self.client.is_layer_present(s2))
        self.assertTrue(self.client.is_layer_present(data))

        # should also work with add_layer
        self.collect.remove(data)
        assert data not in self.collect
        self.assertFalse(self.client.is_layer_present(s1))
        self.collect.append(data)
        self.client.add_layer(data)
        self.assertTrue(self.client.is_layer_present(s1))


    def test_edit_subset_connect_with_data(self):
        data = self.add_data()
        self.assertTrue(self.client.is_layer_present(data.edit_subset))

    def test_edit_subset_removed_with_data(self):
        data = self.add_data()
        self.collect.remove(data)
        self.assertFalse(self.client.is_layer_present(data.edit_subset))

    def test_apply_roi(self):
        data = self.add_data_and_attributes()
        roi = glue.roi.RectangularROI()
        roi.update_limits(.5, .5, 1.5, 1.5)
        x = np.array([1])
        y = np.array([1])
        self.client._apply_roi(roi)
        self.assertTrue(self.layer_data_correct(data.edit_subset, x, y))

    def test_subsets_drawn_over_data(self):
        data = self.add_data_and_attributes()
        subset = data.new_subset()
        self.assertTrue(self.is_first_in_front(subset, data))

if __name__ == "__main__":
    unittest.main()
