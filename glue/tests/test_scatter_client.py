import unittest
from time import sleep

import glue
from glue.scatter_client import ScatterClient

class TestScatterClient(unittest.TestCase):

    def setUp(self):
        self.data = glue.example_data.pipe()[:2]
        self.hub = glue.Hub()
        self.collect = glue.DataCollection()
        self.client = ScatterClient(self.collect)
        self.connect()

    def add_data(self, data=None):
        if data == None:
            data = self.data[0]
        self.collect.append(data)
        self.client.add_layer(data)
        return data

    def add_data_and_attributes(self):
        data = self.add_data()
        self.client.set_xdata('A_Vb')
        self.client.set_ydata('A_Vc')
        return data

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
        self.assertRaises(TypeError, self.client.add_layer, data)

    def test_valid_add(self):
        layer = self.add_data()
        self.assertTrue(self.client.is_layer_present(self.data[0]))

    def test_axis_labels_sync_with_setters(self):
        layer = self.add_data()
        self.client.set_xdata('A_Vc')
        self.assertEquals(self.client.ax.get_xlabel(), 'A_Vc')
        self.client.set_ydata('A_Vb')
        self.assertEquals(self.client.ax.get_ylabel(), 'A_Vb')

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
        self.assertEquals(len(self.client.ax.collections), 1)
        layer = self.add_data()
        self.assertEquals(len(self.client.ax.collections), 1)


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

    def test_add_subset(self):
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
        x = layer['A_Vb']
        y = layer['A_Vc']
        self.assertTrue(self.layer_data_correct(layer, x, y))

    def test_attribute_update_plot_data(self):
        layer = self.add_data_and_attributes()
        x = layer['A_Vb']
        y = layer['A_Vb']
        self.client.set_ydata('A_Vb')
        self.assertTrue(self.layer_data_correct(layer, x, y))

    def test_invalid_plot(self):
        layer = self.add_data_and_attributes()
        self.assertTrue(self.layer_drawn(layer))
        self.client.set_xdata('attribute_does_not_exist')
        self.assertFalse(self.layer_drawn(layer))

    def test_two_incompatible_data(self):
        d0 = self.add_data(self.data[0])
        d1 = self.add_data(self.data[1])
        self.client.set_xdata('A_Vb')
        self.client.set_ydata('A_Vc')
        x = d0['A_Vb']
        y = d0['A_Vc']
        self.assertTrue(self.layer_drawn(d0))
        self.assertTrue(self.layer_data_correct(d0, x, y))
        self.assertFalse(self.layer_drawn(d1))

        self.client.set_xdata('GLON')
        self.client.set_ydata('GLAT')
        x = d1['GLON']
        y = d1['GLAT']
        self.assertTrue(self.layer_drawn(d1))
        self.assertTrue(self.layer_data_correct(d1, x, y))
        self.assertFalse(self.layer_drawn(d0))



if __name__ == "__main__":
    unittest.main()
