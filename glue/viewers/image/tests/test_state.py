from __future__ import absolute_import, division, print_function

import pytest

import numpy as np
from numpy.testing import assert_equal

from glue.core import Data, DataCollection
from glue.core.coordinates import Coordinates
from glue.core.link_helpers import LinkSame
from glue.core.exceptions import IncompatibleDataException, IncompatibleAttribute

from ..state import ImageViewerState, ImageLayerState, AggregateSlice


class SimpleCoordinates(Coordinates):

    def world2pixel(self, *world):
        return tuple([0.4 * w for w in world])

    def pixel2world(self, *pixel):
        return tuple([2.5 * p for p in pixel])

    def dependent_axes(self, axis):
        if axis in (1, 2):
            return (1, 2)
        else:
            return (axis,)


class TestImageViewerState(object):

    def setup_method(self, method):
        self.state = ImageViewerState()

    def test_pixel_world_linking(self):

        data = Data(label='data', x=[[1, 2], [3, 4]], y=[[5, 6], [7, 8]])
        layer_state = ImageLayerState(layer=data, viewer_state=self.state)
        self.state.layers.append(layer_state)

        w1, w2 = data.world_component_ids
        p1, p2 = data.pixel_component_ids

        self.state.reference_data = data

        # Setting world components should set the pixel ones

        self.state.x_att_world = w1
        self.state.y_att_world = w2

        assert self.state.x_att is p1
        assert self.state.y_att is p2

        # Setting one component to the same as the other should trigger the other
        # to flip to prevent them from both being the same

        self.state.x_att_world = w2
        assert self.state.x_att is p2
        assert self.state.y_att is p1
        assert self.state.y_att_world is w1

        self.state.y_att_world = w2
        assert self.state.x_att is p1
        assert self.state.x_att_world is w1
        assert self.state.y_att is p2

        # Changing x_att and y_att should change the world equivalents

        self.state.x_att = p2
        assert self.state.x_att_world is w2
        assert self.state.y_att is p1
        assert self.state.y_att_world is w1

        self.state.y_att = p2
        assert self.state.y_att_world is w2
        assert self.state.x_att is p1
        assert self.state.x_att_world is w1


class TestSlicingAggregation():

    def setup_method(self, method):
        self.viewer_state = ImageViewerState()
        self.data = Data(x=np.ones((3, 4, 5, 6, 7)))
        self.layer_state = ImageLayerState(layer=self.data, viewer_state=self.viewer_state)
        self.viewer_state.layers.append(self.layer_state)
        self.p = self.data.pixel_component_ids

    def test_default(self):
        # Check default settings
        assert self.viewer_state.x_att == self.p[4]
        assert self.viewer_state.y_att == self.p[3]
        assert self.viewer_state.slices == (0, 0, 0, 0, 0)
        assert self.layer_state.get_sliced_data().shape == (6, 7)

    def test_flipped(self):
        # Make sure slice is transposed if needed
        self.viewer_state.x_att = self.p[3]
        self.viewer_state.y_att = self.p[4]
        assert self.viewer_state.slices == (0, 0, 0, 0, 0)
        assert self.layer_state.get_sliced_data().shape == (7, 6)

    def test_slice_preserved(self):
        # Make sure slice stays the same if changing attributes
        self.viewer_state.slices = (1, 3, 2, 5, 4)
        self.viewer_state.x_att = self.p[2]
        self.viewer_state.y_att = self.p[4]
        assert self.viewer_state.slices == (1, 3, 2, 5, 4)
        assert self.viewer_state.wcsaxes_slice == ['y', 5, 'x', 3, 1]
        assert self.layer_state.get_sliced_data().shape == (7, 5)
        self.viewer_state.x_att = self.p[2]
        self.viewer_state.y_att = self.p[1]
        assert self.viewer_state.slices == (1, 3, 2, 5, 4)
        assert self.viewer_state.wcsaxes_slice == [4, 5, 'x', 'y', 1]
        assert self.layer_state.get_sliced_data().shape == (4, 5)
        self.viewer_state.x_att = self.p[0]
        self.viewer_state.y_att = self.p[4]
        assert self.viewer_state.slices == (1, 3, 2, 5, 4)
        assert self.viewer_state.wcsaxes_slice == ['y', 5, 2, 3, 'x']
        assert self.layer_state.get_sliced_data().shape == (7, 3)

    def test_aggregation(self):
        # Check whether using AggregateSlice works
        slc1 = AggregateSlice(slice(None), 0, np.mean)
        slc2 = AggregateSlice(slice(2, 5), 3, np.sum)
        self.viewer_state.slices = (slc1, 3, 2, slc2, 4)
        self.viewer_state.x_att = self.p[2]
        self.viewer_state.y_att = self.p[4]
        assert self.viewer_state.slices == (slc1, 3, 2, slc2, 4)
        assert self.viewer_state.wcsaxes_slice == ['y', 3, 'x', 3, 0]
        result = self.layer_state.get_sliced_data()
        assert result.shape == (7, 5)
        assert_equal(result, 3)  # sum along 3 indices in one of the dimensions


class TestReprojection():

    def setup_method(self, method):

        self.data_collection = DataCollection()

        self.array = np.arange(3024).reshape((6, 7, 8, 9))

        # The reference dataset. Shape is (6, 7, 8, 9).
        self.data1 = Data(x=self.array)
        self.data_collection.append(self.data1)

        # A dataset with the same shape but not linked. Shape is (6, 7, 8, 9).
        self.data2 = Data(x=self.array)
        self.data_collection.append(self.data2)

        # A dataset with the same number of dimesnions but in a different
        # order, linked to the first. Shape is (9, 7, 6, 8).
        self.data3 = Data(x=np.moveaxis(self.array, (3, 1, 0, 2), (0, 1, 2, 3)))
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
        self.data4 = Data(x=self.array[:, :, :, 0].transpose())
        self.data_collection.append(self.data4)
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[0],
                                               self.data4.pixel_component_ids[2]))
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[1],
                                               self.data4.pixel_component_ids[1]))
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[2],
                                               self.data4.pixel_component_ids[0]))

        # A dataset with even fewer dimensions, linked to the first one. Shape
        # is (8, 6)
        self.data5 = Data(x=self.array[:, 0, :, 0].transpose())
        self.data_collection.append(self.data5)
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[0],
                                               self.data5.pixel_component_ids[1]))
        self.data_collection.add_link(LinkSame(self.data1.pixel_component_ids[2],
                                               self.data5.pixel_component_ids[0]))

        # A dataset that is not on the same pixel grid and requires reprojection
        self.data6 = Data()
        self.data6.coords = SimpleCoordinates()
        self.array_nonaligned = np.arange(60).reshape((5, 3, 4))
        self.data6['x'] = np.array(self.array_nonaligned)
        self.data_collection.append(self.data6)
        self.data_collection.add_link(LinkSame(self.data1.world_component_ids[0],
                                               self.data6.world_component_ids[1]))
        self.data_collection.add_link(LinkSame(self.data1.world_component_ids[1],
                                               self.data6.world_component_ids[2]))
        self.data_collection.add_link(LinkSame(self.data1.world_component_ids[2],
                                               self.data6.world_component_ids[0]))

        self.viewer_state = ImageViewerState()
        self.viewer_state.layers.append(ImageLayerState(viewer_state=self.viewer_state, layer=self.data1))
        self.viewer_state.layers.append(ImageLayerState(viewer_state=self.viewer_state, layer=self.data2))
        self.viewer_state.layers.append(ImageLayerState(viewer_state=self.viewer_state, layer=self.data3))
        self.viewer_state.layers.append(ImageLayerState(viewer_state=self.viewer_state, layer=self.data4))
        self.viewer_state.layers.append(ImageLayerState(viewer_state=self.viewer_state, layer=self.data5))
        self.viewer_state.layers.append(ImageLayerState(viewer_state=self.viewer_state, layer=self.data6))

        self.viewer_state.reference_data = self.data1

    def test_default_axis_order(self):

        # Start off with a combination of x/y that means that only one of the
        # other datasets will be matched.

        self.viewer_state.x_att = self.data1.pixel_component_ids[3]
        self.viewer_state.y_att = self.data1.pixel_component_ids[2]
        self.viewer_state.slices = (3, 2, 4, 1)

        image = self.viewer_state.layers[0].get_sliced_data()
        assert_equal(image, self.array[3, 2, :, :])

        with pytest.raises(IncompatibleAttribute):
            self.viewer_state.layers[1].get_sliced_data()

        image = self.viewer_state.layers[2].get_sliced_data()
        assert_equal(image, self.array[3, 2, :, :])

        with pytest.raises(IncompatibleDataException):
            self.viewer_state.layers[3].get_sliced_data()

        with pytest.raises(IncompatibleDataException):
            self.viewer_state.layers[4].get_sliced_data()

    def test_transpose_axis_order(self):

        # Next make it so the x/y axes correspond to the dimensions with length
        # 6 and 8 which most datasets will be compatible with, and this also
        # requires a tranposition.

        self.viewer_state.x_att = self.data1.pixel_component_ids[0]
        self.viewer_state.y_att = self.data1.pixel_component_ids[2]
        self.viewer_state.slices = (3, 2, 4, 1)

        image = self.viewer_state.layers[0].get_sliced_data()
        print(image.shape)
        assert_equal(image, self.array[:, 2, :, 1].transpose())

        with pytest.raises(IncompatibleAttribute):
            self.viewer_state.layers[1].get_sliced_data()

        image = self.viewer_state.layers[2].get_sliced_data()
        print(image.shape)
        assert_equal(image, self.array[:, 2, :, 1].transpose())

        image = self.viewer_state.layers[3].get_sliced_data()
        assert_equal(image, self.array[:, 2, :, 0].transpose())

        image = self.viewer_state.layers[4].get_sliced_data()
        assert_equal(image, self.array[:, 0, :, 0].transpose())

    def test_transpose_axis_order_view(self):

        # As for the previous test, but this time with a view applied

        self.viewer_state.x_att = self.data1.pixel_component_ids[0]
        self.viewer_state.y_att = self.data1.pixel_component_ids[2]
        self.viewer_state.slices = (3, 2, 4, 1)

        view = [slice(1, None, 2), slice(None, None, 3)]

        image = self.viewer_state.layers[0].get_sliced_data(view=view)
        assert_equal(image, self.array[::3, 2, 1::2, 1].transpose())

        with pytest.raises(IncompatibleAttribute):
            self.viewer_state.layers[1].get_sliced_data(view=view)

        image = self.viewer_state.layers[2].get_sliced_data(view=view)
        print(image.shape)
        assert_equal(image, self.array[::3, 2, 1::2, 1].transpose())

        image = self.viewer_state.layers[3].get_sliced_data(view=view)
        assert_equal(image, self.array[::3, 2, 1::2, 0].transpose())

        image = self.viewer_state.layers[4].get_sliced_data(view=view)
        assert_equal(image, self.array[::3, 0, 1::2, 0].transpose())

    def test_reproject(self):

        # Test a case where the data needs to actually be reprojected

        # As for the previous test, but this time with a view applied

        self.viewer_state.x_att = self.data1.pixel_component_ids[0]
        self.viewer_state.y_att = self.data1.pixel_component_ids[2]
        self.viewer_state.slices = (3, 2, 4, 1)

        view = [slice(1, None, 2), slice(None, None, 3)]

        actual = self.viewer_state.layers[5].get_sliced_data(view=view)

        # The data to be reprojected is 3-dimensional. The axes we have set
        # correspond to 1 (for x) and 0 (for y). The third dimension of the
        # data to be reprojected should be sliced. This is linked with the
        # second dimension of the original data, for which the slice index is
        # 2. Since the data to be reprojected has coordinates that are 2.5 times
        # those of the reference data, this means the slice index should be 0.8,
        # which rounded corresponds to 1.
        expected = self.array_nonaligned[:, :, 1]

        # Now in the frame of the reference data, the data to show are indices
        # [0, 3] along x and [1, 3, 5, 7] along y. Applying the transformation,
        # this gives values of [0, 1.2] and [0.4, 1.2, 2, 2.8] for x and y,
        # and rounded, this gives [0, 1] and [0, 1, 2, 3]. As a reminder, in the
        # data to reproject, dimension 0 is y and dimension 1 is x
        expected = expected[:4, :2]

        # Let's make sure this works!
        assert_equal(actual, expected)

    def test_too_many_dimensions(self):

        # If we change the reference data, then the first dataset won't be
        # visible anymore because it has too many dimensions

        self.viewer_state.reference_data = self.data4

        with pytest.raises(IncompatibleAttribute):
            self.viewer_state.layers[0].get_sliced_data()

        self.viewer_state.reference_data = self.data6

        with pytest.raises(IncompatibleAttribute):
            self.viewer_state.layers[0].get_sliced_data()

    def test_reset_pixel_cache(self):

        # Test to make sure that resetting the pixel cache works properly

        self.viewer_state.x_att = self.data1.pixel_component_ids[0]
        self.viewer_state.y_att = self.data1.pixel_component_ids[2]

        self.viewer_state.slices = (1, 1, 1, 1)

        layer = self.viewer_state.layers[5]

        assert layer._pixel_cache is None

        layer.get_sliced_data()

        assert layer._pixel_cache['reset_slices'][0] == [False, False, True, False]
        assert layer._pixel_cache['reset_slices'][1] == [True, True, False, False]
        assert layer._pixel_cache['reset_slices'][2] == [True, True, False, False]

        self.viewer_state.slices = (1, 1, 1, 2)

        assert layer._pixel_cache['reset_slices'][0] == [False, False, True, False]
        assert layer._pixel_cache['reset_slices'][1] == [True, True, False, False]
        assert layer._pixel_cache['reset_slices'][2] == [True, True, False, False]

        self.viewer_state.slices = (1, 2, 1, 2)

        assert layer._pixel_cache['reset_slices'][0] == [False, False, True, False]
        assert layer._pixel_cache['reset_slices'][1] is None
        assert layer._pixel_cache['reset_slices'][2] is None

        self.viewer_state.slices = (1, 2, 2, 2)

        assert layer._pixel_cache['reset_slices'][0] is None
        assert layer._pixel_cache['reset_slices'][1] is None
        assert layer._pixel_cache['reset_slices'][2] is None
