from unittest.mock import MagicMock
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
        self.default_bounds = [(0, 1, 2), (0, 1, 2)]

    def test_shape_size_ndim_dtype(self):

        assert self.composite.shape is None
        assert self.composite.size is None
        assert self.composite.ndim == 2  # for now, this is hard-coded
        assert self.composite.dtype is np.dtype(float)  # for now, this is hard-coded

        self.composite.allocate('a')
        self.composite.set('a', array=self.array1)

        assert self.composite.shape == (2, 2)
        assert self.composite.size == 4
        assert self.composite.ndim == 2
        assert self.composite.dtype is np.dtype(float)

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

        self.composite.mode = 'colormap'

        self.composite.allocate('a')
        self.composite.allocate('b')

        self.composite.set('a', zorder=0, visible=True, array=self.array1,
                           cmap=cm.Blues, clim=(0, 2))

        self.composite.set('b', zorder=1, visible=True, array=self.array2,
                           cmap=cm.Reds, clim=(0, 1))

        # Determine expected result for each layer individually in the absence
        # of transparency

        expected_a = np.array([[cm.Blues(1.), cm.Blues(0.5)],
                               [cm.Blues(0.), cm.Blues(0.)]])

        if self.composite.cmap_bad is None:
            # default masked values are opaque black
            bad_rgba = (0, 0, 0, 1)
        else:
            # otherwise, apply cmap_bad_alpha:
            bad_rgba = cm.Reds.get_bad()[:3] + (self.composite.cmap_bad_alpha,)

        expected_b = np.array(
            [[bad_rgba, cm.Reds(1.)],
             [cm.Reds(0.), cm.Reds(0.)]]
        )

        # If both layers have alpha=1, the top layer should be the only one visible
        assert_allclose(self.composite(bounds=self.default_bounds), expected_b)

        # If the top layer has alpha=0, the bottom layer should be the only one visible
        self.composite.set('b', alpha=0.)

        assert_allclose(self.composite(bounds=self.default_bounds), expected_a)

        # If the top layer has alpha=0.5, the result should be an equal blend of each

        self.composite.set('b', alpha=0.5)

        assert_allclose(self.composite(bounds=self.default_bounds), 0.5 * (expected_b + expected_a))

    def test_cmap_alphas(self):

        self.composite.mode = 'colormap'

        self.composite.allocate('a')
        self.composite.allocate('b')

        self.composite.set('a', zorder=0, visible=True, array=self.array1,
                           cmap=cm.Blues, clim=(0, 2))

        self.composite.set('b', zorder=1, visible=True, array=self.array2,
                           cmap=lambda x: cm.Reds(x, alpha=abs(x)), clim=(0, 1))

        # Determine expected result for each layer individually in the absence
        # of transparency

        expected_a = np.array([[cm.Blues(1.), cm.Blues(0.5)],
                               [cm.Blues(0.), cm.Blues(0.)]])

        bad_value = cm.Reds.with_extremes(
            # set "bad" to the default color and alpha=0
            bad=cm.Reds.get_bad().tolist()[:3] + [0.]
        )(np.nan)
        expected_b_1 = np.array([[bad_value, cm.Reds(1.)],
                                 [cm.Reds(0.), cm.Reds(0.)]])

        # If the top layer has alpha=1 with a colormap alpha fading proportional to absval,
        # it should be visible only at the nonzero value [0, 1]

        assert_allclose(self.composite(bounds=self.default_bounds),
                        [[expected_a[0, 0], expected_b_1[0, 1]], expected_a[1]])

        # For the same case with the top layer alpha=0.5 that value should become an equal
        # blend of both layers again

        self.composite.set('b', alpha=0.5)

        assert_allclose(self.composite(bounds=self.default_bounds),
                        [[expected_a[0, 0], 0.5 * (expected_a[0, 1] + expected_b_1[0, 1])],
                         expected_a[1]])

        # A third layer added at the bottom should not be visible in the output

        self.composite.allocate('c')
        self.composite.set('c', zorder=-1, visible=True, array=self.array3,
                           cmap=cm.Greens, clim=(0, 2))

        assert_allclose(self.composite(bounds=self.default_bounds),
                        [[expected_a[0, 0], 0.5 * (expected_a[0, 1] + expected_b_1[0, 1])],
                         expected_a[1]])

        # For only the bottom layer having such colormap, the top layer should appear just the same

        self.composite.set('a', alpha=1., cmap=lambda x: cm.Blues(x, alpha=abs(x)))
        self.composite.set('b', alpha=1., cmap=cm.Reds)

        bad_value = cm.Reds.with_extremes(
            # set "bad" to the default color and alpha=1
            bad=cm.Reds.get_bad().tolist()[:3] + [1.]
        )(np.nan)
        expected_b_2 = np.array([[bad_value, cm.Reds(1.)],
                                 [cm.Reds(0.), cm.Reds(0.)]])
        assert_allclose(self.composite(bounds=self.default_bounds), expected_b_2)

        # Setting the third layer on top with alpha=0 should not affect the appearance

        self.composite.set('c', zorder=2, alpha=0.)

        assert_allclose(self.composite(bounds=self.default_bounds), expected_b_2)

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

        assert_allclose(self.composite(bounds=self.default_bounds), np.maximum(expected_a, expected_b))

        # If the top layer has alpha=0, the bottom layer should be the only one visible

        self.composite.set('b', alpha=0.)

        assert_allclose(self.composite(bounds=self.default_bounds), expected_a)

        # If the top layer has alpha=0.5, the result should have

        self.composite.set('b', alpha=0.5)

        assert_allclose(self.composite(bounds=self.default_bounds), np.maximum(expected_a, expected_b * 0.5))

    def test_deallocate(self):

        self.composite.allocate('a')
        self.composite.set('a', array=self.array1, color='1.0', clim=(0, 2))

        assert self.composite.shape == (2, 2)
        expected = np.ones(4) * self.array1[:, :, np.newaxis] / 2.
        expected[:, :, 3] = 1
        assert_allclose(self.composite(bounds=self.default_bounds), expected)

        self.composite.deallocate('a')
        assert self.composite.shape is None
        assert self.composite(bounds=self.default_bounds) is None

    def test_noactive(self):

        array = np.random.random((100, 100))

        self.composite.allocate('a')
        self.composite.set('a', array=array, visible=False)

        assert self.composite() is None

    def test_cmap_bad(self):
        self.composite.mode = 'colormap'

        self.composite.allocate('a')
        self.composite.allocate('b')
        cmap_a = cm.viridis
        cmap_b = cm.Grays

        self.composite.set('a', zorder=0, visible=True, array=self.array1,
                           cmap=cmap_a, clim=(0, 2))

        composite_arr_a = self.composite(bounds=self.default_bounds)
        self.composite.set('b', zorder=1, visible=True, array=self.array2 * np.nan,
                           cmap=cmap_b, clim=(0, 1))

        shape = self.array2.shape
        bad_color = tuple(cmap_b.get_bad()[:3])
        bad_expected = {
            # masked pixels are opaque black - this is the glue
            # default behavior.
            None: np.broadcast_to((0, 0, 0, 1,), (*shape, 4)),

            # masked pixels are totally transparent, the composite
            # image is simply the lower layer:
            (0, 0, 0, 0): composite_arr_a,

            # masked pixels are set to the "bad" color with alpha==1
            (0, 0, 0, 1): np.broadcast_to(bad_color + (1,), (*shape, 4)),
        }

        for cmap_bad, expected in bad_expected.items():
            self.composite.cmap_bad = cmap_bad
            self.composite.set('b', cmap_bad=cmap_bad)
            actual = self.composite(bounds=self.default_bounds)
            assert_allclose(actual, expected)
