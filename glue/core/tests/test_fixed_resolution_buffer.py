from __future__ import absolute_import, division, print_function

import pytest

import numpy as np
from numpy.testing import assert_equal

from glue.tests.helpers import HYPOTHESIS_INSTALLED
from glue.core import Data, DataCollection
from glue.core.link_helpers import LinkSame
from glue.core.exceptions import IncompatibleDataException, IncompatibleAttribute

from glue.core.fixed_resolution_buffer import split_view_for_bounds


ARRAY = np.arange(3024).reshape((6, 7, 8, 9))


CASES = [
    ((10,), [(0, 10, 1)], (10,), (slice(0, 10),), (slice(0, 10, 1),)),
    ((10,), [(3, 6, 1)], (3,), (slice(0, 3,),), (slice(3, 6, 1),)),
    ((10,), [(3, 13, 1)], (10,), (slice(0, 7),), (slice(3, 10, 1),)),
    ((10,), [(-5, 15, 1)], (20,), (slice(5, 15),), (slice(0, 10, 1),)),
    ((10,), [(-10, -5, 1)], (5,), None, None),
    ((10,), [(15, 20, 1)], (5,), None, None),
    ((10,), [(0, 10, 2)], (5,), (slice(0, 5),), (slice(0, 10, 2),)),
    ((10,), [(0, 15, 2)], (8,), (slice(0, 5),), (slice(0, 10, 2),)),
    ((10,), [(-5, 15, 2)], (10,), (slice(3, 8),), (slice(1, 10, 2),)),
    ((10,), [(-1, 15, 5)], (4,), (slice(1, 3),), (slice(4, 10, 5),)),
    ((10,), [(-100, 1000, 200)], (6,), None, None),
    ((10,), [(-100, 1000, 26)], (43,), (slice(4, 5),), (slice(4, 10, 26),))]


@pytest.mark.parametrize(('data_shape', 'bounds', 'buffer_shape', 'buffer_view', 'data_view'), CASES)
def test_split_view_for_bounds(data_shape, bounds, buffer_shape, buffer_view, data_view):
    bs, bv, dv = split_view_for_bounds(data_shape, bounds)
    assert bs == buffer_shape
    assert bv == buffer_view
    assert dv == data_view


if HYPOTHESIS_INSTALLED:

    from hypothesis import given, settings, example
    from hypothesis.strategies import none, integers

    @given(size=integers(1, 100),
           beg=integers(-100, 100),
           end=integers(1, 100),
           stp=integers(1, 100))
    @settings(max_examples=10000, derandomize=True)
    @example(size=2, beg=0, end=1, stp=2)
    @example(size=1, beg=1, end=1, stp=1)
    def test_split_view_for_bounds_hypot(size, beg, end, stp):

        # Make sure end > beg
        end = end + beg

        bs, bv, dv = split_view_for_bounds((size,), [(beg, end, stp)])

        # We need to do things differently here than in the function otherwise
        # it defies the point of the test. So to do this, we check what happens
        # if we use the above values on example arrays.
        input_array = np.random.random(size)
        buffer = np.zeros(bs)
        if dv is not None:
            buffer[bv] = input_array[dv]

        # Now do the same but using the original slice, on a padded array
        # without using the results from split_view_for_bounds
        pad = 300
        padded_array = np.pad(input_array, pad, mode='constant', constant_values=0)
        expected = padded_array[slice(beg + pad, end + pad, stp)]

        assert_equal(buffer, expected)


class TestFixedResolutionBuffer():

    def setup_method(self, method):

        self.data_collection = DataCollection()

        # The reference dataset. Shape is (6, 7, 8, 9).
        self.data1 = Data(x=ARRAY)
        self.data_collection.append(self.data1)

        # A dataset with the same shape but not linked. Shape is (6, 7, 8, 9).
        self.data2 = Data(x=ARRAY)
        self.data_collection.append(self.data2)

        # A dataset with the same number of dimensions but in a different
        # order, linked to the first. Shape is (9, 7, 6, 8).
        self.data3 = Data(x=np.moveaxis(ARRAY, (3, 1, 0, 2), (0, 1, 2, 3)))
        self.data_collection.append(self.data3)
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[0],
                                               self.data3.pixel_component_ids[2]))
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[1],
                                               self.data3.pixel_component_ids[1]))
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[2],
                                               self.data3.pixel_component_ids[3]))
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[3],
                                               self.data3.pixel_component_ids[0]))

        # A dataset with fewer dimensions, linked to the first one. Shape is
        # (8, 7, 6)
        self.data4 = Data(x=ARRAY[:, :, :, 0].transpose())
        self.data_collection.append(self.data4)
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[0],
                                               self.data4.pixel_component_ids[2]))
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[1],
                                               self.data4.pixel_component_ids[1]))
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[2],
                                               self.data4.pixel_component_ids[0]))

        # A dataset with even fewer dimensions, linked to the first one. Shape
        # is (8, 6)
        self.data5 = Data(x=ARRAY[:, 0, :, 0].transpose())
        self.data_collection.append(self.data5)
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[0],
                                               self.data5.pixel_component_ids[1]))
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[2],
                                               self.data5.pixel_component_ids[0]))

        # A dataset that is not on the same pixel grid and requires reprojection
        # self.data6 = Data()
        # self.data6.coords = SimpleCoordinates()
        # self.array_nonaligned = np.arange(60).reshape((5, 3, 4))
        # self.data6['x'] = np.array(self.array_nonaligned)
        # self.data_collection.append(self.data6)
        # self.data_collection.add_link(LinkSame(self.data1.world_component_ids[0],
        #                                        self.data6.world_component_ids[1]))
        # self.data_collection.add_link(LinkSame(self.data1.world_component_ids[1],
        #                                        self.data6.world_component_ids[2]))
        # self.data_collection.add_link(LinkSame(self.data1.world_component_ids[2],
        #                                        self.data6.world_component_ids[0]))

    # Start off with the cases where the data is the target data. Enumerate
    # the different cases for the bounds and the expected result.

    DATA_IS_TARGET_CASES = [

        # Bounds are full extent of data
        (((0, 6, 1), (0, 7, 1), (0, 8, 1), (0, 9, 1)),
            ARRAY),

        # Bounds are inside data
        (((2, 4, 1), (3, 4, 1), (0, 8, 1), (0, 8, 1)),
            ARRAY[2:4, 3:4, :, :8]),

        # Bounds are outside data along some dimensions
        (((-5, 10, 1), (3, 6, 1), (0, 10, 1), (5, 7, 1)),
            np.pad(ARRAY[:, 3:6, :, 5:7], [(5, 4), (0, 0), (0, 2), (0, 0)],
                   mode='constant', constant_values=0)),

        # No overlap
        (((2, 4, 1), (3, 4, 1), (-5, -3, 1), (0, 8, 1)),
            np.zeros((2, 1, 2, 8)))

    ]

    @pytest.mark.parametrize(('bounds', 'expected'), DATA_IS_TARGET_CASES)
    def test_data_is_target_full_bounds(self, bounds, expected):

        buffer = self.data1.get_fixed_resolution_buffer(self.data1, bounds,
                                                        target_cid=self.data1.id['x'])
        assert_equal(buffer, expected)

        buffer = self.data3.get_fixed_resolution_buffer(self.data1, bounds,
                                                        target_cid=self.data3.id['x'])
        assert_equal(buffer, expected)


# def test_reset_pixel_cache(self):
#
#     # Test to make sure that resetting the pixel cache works properly
#
#     self.viewer_state.x_att = self.data1.pixel_component_ids[0]
#     self.viewer_state.y_att = self.data1.pixel_component_ids[2]
#
#     self.viewer_state.slices = (1, 1, 1, 1)
#
#     layer = self.viewer_state.layers[5]
#
#     assert layer._pixel_cache is None
#
#     layer.get_sliced_data()
#
#     assert layer._pixel_cache['reset_slices'][0] == [False, False, True, False]
#     assert layer._pixel_cache['reset_slices'][1] == [True, True, False, False]
#     assert layer._pixel_cache['reset_slices'][2] == [True, True, False, False]
#
#     self.viewer_state.slices = (1, 1, 1, 2)
#
#     assert layer._pixel_cache['reset_slices'][0] == [False, False, True, False]
#     assert layer._pixel_cache['reset_slices'][1] == [True, True, False, False]
#     assert layer._pixel_cache['reset_slices'][2] == [True, True, False, False]
#
#     self.viewer_state.slices = (1, 2, 1, 2)
#
#     assert layer._pixel_cache['reset_slices'][0] == [False, False, True, False]
#     assert layer._pixel_cache['reset_slices'][1] is None
#     assert layer._pixel_cache['reset_slices'][2] is None
#
#     self.viewer_state.slices = (1, 2, 2, 2)
#
#     assert layer._pixel_cache['reset_slices'][0] is None
#     assert layer._pixel_cache['reset_slices'][1] is None
#     assert layer._pixel_cache['reset_slices'][2] is None
