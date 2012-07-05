import pytest

import matplotlib.pyplot as plt
from mock import MagicMock
import numpy as np

from ...tests import example_data
from ... import core

from ..image_client import ImageClient

# share matplotlib instance, and disable rendering, for speed
FIGURE = plt.figure()
AXES = FIGURE.add_subplot(111)
FIGURE.canvas.draw = lambda: 0
plt.close('all')


class DummyCoords(core.coordinates.Coordinates):
    def pixel2world(self, *args):
        result = []
        for i, a in enumerate(args):
            result.append([aa * (i + 1) for aa in a])
        return result


class TestImageClient(object):

    def setup_method(self, method):
        self.im = example_data.test_image()
        self.cube = example_data.test_cube()
        self.collect = core.data_collection.DataCollection()

    def create_client_with_image(self):
        client = ImageClient(self.collect, axes=AXES)
        self.collect.append(self.im)
        client.set_data(self.im)
        return client

    def create_client_with_cube(self):
        client = ImageClient(self.collect, axes=AXES)
        self.collect.append(self.cube)
        client.set_data(self.cube)
        return client

    def test_empty_creation(self):
        client = ImageClient(self.collect, axes=AXES)
        assert client.display_data is None

    def test_nonempty_creation(self):
        self.collect.append(self.im)
        client = ImageClient(self.collect, axes=AXES)
        assert client.display_data is None
        assert not self.im in client.layers

    def test_invalid_add(self):
        client = ImageClient(self.collect, axes=AXES)
        with pytest.raises(TypeError):
            client.add_layer(self.cube)

    def test_set_data(self):
        client = self.create_client_with_image()
        assert client.display_data is self.im

    def test_slice_disabled_for_2d(self):
        client = self.create_client_with_image()
        assert client.slice_ind is None
        with pytest.raises(IndexError):
            client.slice_ind = 10

    def test_slice_disabled_for_no_data(self):
        client = ImageClient(self.collect, axes=AXES)
        assert client.slice_ind is None
        with pytest.raises(IndexError):
            client.slice_ind = 10

    def test_slice_enabled_for_3D(self):
        client = self.create_client_with_cube()
        assert client.slice_ind is not None
        client.slice_ind = 5
        assert client.slice_ind == 5

    def test_add_subset_via_method(self):
        client = ImageClient(self.collect, axes=AXES)
        self.collect.append(self.im)
        s = self.im.new_subset()
        client.add_layer(s)
        assert s in client.layers

    def test_remove_data(self):
        client = ImageClient(self.collect, axes=AXES)
        self.collect.append(self.im)
        s = self.im.new_subset()
        client.add_layer(self.im)
        assert self.im in client.layers
        assert s in client.layers
        client.delete_layer(self.im)
        assert client.display_data is None
        assert not self.im in client.layers
        assert not s in client.layers

    def test_set_norm(self):
        client = self.create_client_with_image()
        assert client.display_data is not None
        client.set_norm(vmin=10, vmax=100)
        assert client.layers[self.im].norm.vmin == 10
        assert client.layers[self.im].norm.vmax == 100

    def test_delete_data(self):
        client = self.create_client_with_image()
        client.delete_layer(self.im)
        assert not self.im in client.layers

    def test_set_attribute(self):
        client = self.create_client_with_image()
        atts = self.im.component_ids()
        assert len(atts) > 1
        for att in atts:
            client.set_attribute(att)
            assert client.display_attribute is att

    def test_set_data_and_attribute(self):
        client = self.create_client_with_image()
        atts = self.im.component_ids()
        assert len(atts) > 1
        for att in atts:
            client.set_data(self.im, attribute=att)
            assert client.display_attribute is att
            assert client.display_data is self.im

    def test_set_slice(self):
        client = self.create_client_with_image()
        with pytest.raises(IndexError):
            client.slice_ind = 10

    def test_slice_bounds_2d(self):
        client = self.create_client_with_image()
        assert client.slice_bounds() == (0, 0)

    def test_slice_bounds_3d(self):
        client = self.create_client_with_cube()
        shape = self.cube.shape
        assert client.slice_bounds() == (0, shape[2] - 1)
        client.set_slice_ori(0)
        assert client.slice_bounds() == (0, shape[0] - 1)
        client.set_slice_ori(1)
        assert client.slice_bounds() == (0, shape[1] - 1)
        client.set_slice_ori(2)
        assert client.slice_bounds() == (0, shape[2] - 1)

    def test_slice_ori_on_2d_raises(self):
        client = self.create_client_with_image()
        with pytest.raises(IndexError):
            client.set_slice_ori(0)

    def test_slice_ori_out_of_bounds(self):
        client = self.create_client_with_image()
        self.collect.append(self.cube)
        client.set_data(self.cube)
        with pytest.raises(TypeError):
            client.set_slice_ori(100)

    def test_apply_roi_2d(self):
        client = self.create_client_with_image()
        self.collect.append(self.cube)
        client.add_layer(self.cube)
        roi = core.roi.PolygonalROI(vx=[10, 20, 20, 10],
                                    vy=[10, 10, 20, 20])
        client._apply_roi(roi)
        roi2 = self.im.edit_subset.subset_state.roi
        state = self.im.edit_subset.subset_state

        assert roi2.to_polygon()[0] == roi.to_polygon()[0]
        assert roi2.to_polygon()[1] == roi.to_polygon()[1]
        assert state.xatt is self.im.get_pixel_component_id(1)
        assert state.yatt is self.im.get_pixel_component_id(0)

        # subset only applied to active data
        roi3 = self.cube.edit_subset.subset_state
        assert not isinstance(roi3, core.subset.RoiSubsetState)

    def test_apply_roi_3d(self):
        client = self.create_client_with_cube()
        self.cube.coords = DummyCoords()
        roi = core.roi.PolygonalROI(vx=[10, 20, 20, 10],
                                    vy=[10, 10, 20, 20])

        client.set_slice_ori(0)
        client._apply_roi(roi)
        state = self.cube.edit_subset.subset_state
        roi2 = state.roi
        assert state.xatt is self.cube.get_pixel_component_id(2)
        assert state.yatt is self.cube.get_pixel_component_id(1)
        assert roi2.to_polygon()[0] == roi.to_polygon()[0]
        assert roi2.to_polygon()[1] == roi.to_polygon()[1]

        client.set_slice_ori(1)
        client._apply_roi(roi)
        state = self.cube.edit_subset.subset_state
        roi2 = state.roi
        assert state.xatt is self.cube.get_pixel_component_id(2)
        assert state.yatt is self.cube.get_pixel_component_id(0)
        assert roi2.to_polygon()[0] == roi.to_polygon()[0]
        assert roi2.to_polygon()[1] == roi.to_polygon()[1]

        client.set_slice_ori(2)
        client._apply_roi(roi)
        state = self.cube.edit_subset.subset_state
        roi2 = state.roi
        assert state.xatt is self.cube.get_pixel_component_id(1)
        assert state.yatt is self.cube.get_pixel_component_id(0)
        assert roi2.to_polygon()[0] == roi.to_polygon()[0]
        assert roi2.to_polygon()[1] == roi.to_polygon()[1]

    def test_update_subset_zeros_mask_on_error(self):
        client = self.create_client_with_image()
        sub = self.im.edit_subset

        bad_state = MagicMock()
        err = core.exceptions.IncompatibleAttribute("Can't make mask")
        bad_state.to_mask.side_effect = err
        bad_state.to_index_list.side_effect = err
        sub.subset_state = bad_state

        client.layers[sub].mask = np.ones(self.im.shape, dtype=bool)
        client._update_subset_single(sub)
        assert not client.layers[sub].mask.any()

    def test_subsets_shown_on_init(self):
        client = self.create_client_with_image()
        subset = self.im.edit_subset
        manager = client.layers[subset]
        assert manager.artist is not None
        assert manager.is_visible()
