import unittest

import matplotlib.pyplot as plt

import glue
from glue.image_client import ImageClient

class TestImageClient(unittest.TestCase):
    def setUp(self):
        self.im = glue.example_data.test_image()
        self.cube = glue.example_data.test_cube()
        self.collect = glue.DataCollection()

    def tearDown(self):
        plt.close('all')

    def test_empty_creation(self):
        client = ImageClient(self.collect)
        self.assertIsNone(client.display_data)

    def test_nonempty_creation(self):
        self.collect.append(self.im)
        self.collect.append(self.cube)
        client = ImageClient(self.collect)
        self.assertIsNone(client.display_data)
        self.assertFalse(self.im in client.layers)

    def test_set_data(self):
        self.collect.append(self.im)
        client = ImageClient(self.collect)
        self.assertIsNone(client.display_data)
        client.set_data(self.im)
        self.assertIs(client.display_data, self.im)

    def test_slice_disabled_for_2d(self):
        self.collect.append(self.im)
        client = ImageClient(self.collect)
        client.set_data(self.im)
        self.assertIsNone(client.slice_ind)
        self.assertRaises(IndexError, client.slice_ind, 10)

    def test_slice_disabled_for_no_data(self):
        client = ImageClient(self.collect)
        self.assertIsNone(client.slice_ind)
        self.assertRaises(IndexError, client.slice_ind, 10)

    def test_slice_enabled_for_3D(self):
        client = ImageClient(self.collect)
        self.collect.append(self.cube)
        client.set_data(self.cube)
        self.assertIsNotNone(client.slice_ind)
        client.slice_ind = 5
        self.assertEquals(client.slice_ind, 5)

    def test_add_subset_via_method(self):
        client = ImageClient(self.collect)
        self.collect.append(self.im)
        s = self.im.create_subset()
        client.add_layer(s)
        self.assertTrue(s in client.layers)

    def test_remove_data(self):
        client = ImageClient(self.collect)
        self.collect.append(self.im)
        s = self.im.create_subset()
        client.add_layer(self.im)
        self.assertTrue(self.im in client.layers)
        self.assertTrue(s in client.layers)
        client.delete_layer(self.im)
        self.assertIsNone(client.display_data)
        self.assertFalse(self.im in client.layers)
        self.assertFalse(s in client.layers)


if __name__ == "__main__":
    unittest.main(failfast=True)