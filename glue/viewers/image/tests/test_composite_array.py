from __future__ import absolute_import, division, print_function

from mock import MagicMock
import numpy as np
from numpy.testing import assert_allclose
from matplotlib.pyplot import cm

from ..composite_array import CompositeArray


class TestCompositeArray(object):

    def setup_method(self, method):
        self.array1 = np.array([[2.0, 1.0], [0.0, 0.0]])
        self.array2 = np.array([[np.nan, 1.0], [0.0, 0.0]])
        self.array3 = np.array([[0.0, 0.0], [1.0, 0.0]])
        self.array4 = np.array([[0.0, 0.0], [0.0, 1.0]])
        self.composite = CompositeArray()

    def test_shape_size_ndim_dtype(self):

        assert self.composite.shape is None
        assert self.composite.size is None
        assert self.composite.ndim == 2  # for now, this is hard-coded
        assert self.composite.dtype is np.float  # for now, this is hard-coded

        self.composite.allocate('a')
        self.composite.set('a', array=self.array1)

        assert self.composite.shape == (2, 2)
        assert self.composite.size == 4
        assert self.composite.ndim == 2
        assert self.composite.dtype is np.float

    def test_shape_function(self):

        array_func = MagicMock()
        array_func.return_value = None

        assert self.composite.shape is None

        self.composite.allocate('a')
        self.composite.set('a', array=array_func)

        assert self.composite.shape is None

        array_func.return_value = self.array1

        assert self.composite.shape == (2, 2)

    def test_cmap_blending(self):

        self.composite.allocate('a')
        self.composite.allocate('b')

        self.composite.set('a', zorder=0, visible=True, array=self.array1,
                           color=cm.Blues, clim=(0, 2))

        self.composite.set('b', zorder=1, visible=True, array=self.array2,
                           color=cm.Reds, clim=(0, 1))

        # Determine expected result for each layer individually in the absence
        # of transparency

        expected_a = np.array([[cm.Blues(1.), cm.Blues(0.5)],
                               [cm.Blues(0.), cm.Blues(0.)]])

        expected_b = np.array([[cm.Reds(0.), cm.Reds(1.)],
                               [cm.Reds(0.), cm.Reds(0.)]])

        # If both layers have alpha=1, the top layer should be the only one visible

        assert_allclose(self.composite[...], expected_b)

        # If the top layer has alpha=0, the bottom layer should be the only one visible

        self.composite.set('b', alpha=0.)

        assert_allclose(self.composite[...], expected_a)

        # If the top layer has alpha=0.5, the result should be an equal blend of each

        self.composite.set('b', alpha=0.5)

        assert_allclose(self.composite[...], 0.5 * (expected_b + expected_a))

    def test_color_blending(self):

        self.composite.allocate('a')
        self.composite.allocate('b')

        self.composite.set('a', zorder=0, visible=True, array=self.array1,
                           color=(0, 0, 1, 1), clim=(0, 2))

        self.composite.set('b', zorder=1, visible=True, array=self.array2,
                           color=(1, 0, 0, 1), clim=(0, 1))

        # Determine expected result for each layer individually in the absence
        # of transparency

        expected_a = np.array([[(0, 0, 1, 1), (0, 0, 0.5, 1)],
                               [(0, 0, 0, 1), (0, 0, 0, 1)]])

        expected_b = np.array([[(0, 0, 0, 1), (1, 0, 0, 1)],
                               [(0, 0, 0, 1), (0, 0, 0, 1)]])

        # In this mode, the zorder shouldn't matter, and if both layers have
        # alpha=1, we should see a normal blend of the colors

        assert_allclose(self.composite[...], np.maximum(expected_a, expected_b))

        # If the top layer has alpha=0, the bottom layer should be the only one visible

        self.composite.set('b', alpha=0.)

        assert_allclose(self.composite[...], expected_a)

        # If the top layer has alpha=0.5, the result should have

        self.composite.set('b', alpha=0.5)

        assert_allclose(self.composite[...], np.maximum(expected_a, expected_b * 0.5))

    def test_deallocate(self):

        self.composite.allocate('a')
        self.composite.set('a', array=self.array1, color='1.0', clim=(0, 2))

        assert self.composite.shape == (2, 2)
        expected = np.ones(4) * self.array1[:, :, np.newaxis] / 2.
        expected[:, :, 3] = 1
        assert_allclose(self.composite[...], expected)

        self.composite.deallocate('a')
        assert self.composite.shape is None
        assert self.composite[...] is None

    def test_getitem_noactive(self):

        # Regression test for a bug that caused __getitem__ to return an array
        # with the wrong size if a view was used and no layers were active.

        array = np.random.random((100, 100))

        self.composite.allocate('a')
        self.composite.set('a', array=array, visible=False)

        assert self.composite[:].shape == (100, 100, 4)
        assert self.composite[10:90, ::10].shape == (80, 10, 4)
