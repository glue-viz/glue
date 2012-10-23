#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
import pytest

import matplotlib.pyplot as plt
from mock import MagicMock
import numpy as np

from ...tests import example_data
from ... import core
from ...core.exceptions import IncompatibleAttribute

from ..image_client import ImageClient

# share matplotlib instance, and disable rendering, for speed
FIGURE = plt.figure()
FIGURE.canvas.draw = lambda: 0
plt.close('all')


class DummyCoords(core.coordinates.Coordinates):
    def pixel2world(self, *args):
        return tuple(a * (i + 1) for i, a in enumerate(args))


class TrueState(core.subset.SubsetState):
    def to_mask(self, view=None):
        data = np.ones(self.parent.data.shape, dtype=bool)
        if view is not None:
            data = data[view]
        return data


class TestImageClient(object):

    def setup_method(self, method):
        self.im = example_data.test_image()
        self.cube = example_data.test_cube()
        self.scatter = core.Data(x=[1, 2, 3, 4], y=[4, 5, 6, 7])
        self.im.edit_subset = self.im.new_subset()
        self.cube.edit_subset = self.cube.new_subset()
        self.collect = core.data_collection.DataCollection()

    def create_client_with_image(self):
        client = ImageClient(self.collect, figure=FIGURE)
        self.collect.append(self.im)
        client.set_data(self.im)
        return client

    def create_client_with_image_and_scatter(self):
        from glue.core.link_helpers import LinkSame

        client = self.create_client_with_image()
        self.collect.append(self.scatter)

        ix = self.im.get_world_component_id(0)
        iy = self.im.get_world_component_id(1)

        self.collect.add_link(LinkSame(self.scatter.id['x'], ix))
        self.collect.add_link(LinkSame(self.scatter.id['x'], iy))
        client.add_scatter_layer(self.scatter)

        return client

    def create_client_with_cube(self):
        client = ImageClient(self.collect, figure=FIGURE)
        self.collect.append(self.cube)
        client.set_data(self.cube)
        return client

    def test_empty_creation(self):
        client = ImageClient(self.collect, figure=FIGURE)
        assert client.display_data is None

    def test_nonempty_creation(self):
        self.collect.append(self.im)
        client = ImageClient(self.collect, figure=FIGURE)
        assert client.display_data is None
        assert not self.im in client.artists

    def test_invalid_add(self):
        client = ImageClient(self.collect, figure=FIGURE)
        with pytest.raises(TypeError) as exc:
            client.add_layer(self.cube)
        assert exc.value.args[0] == ("Data not managed by client's "
                                     "data collection")

    def test_set_data(self):
        client = self.create_client_with_image()
        assert client.display_data is self.im

    def test_slice_disabled_for_2d(self):
        client = self.create_client_with_image()
        assert client.slice_ind is None
        with pytest.raises(IndexError) as exc:
            client.slice_ind = 10
        assert exc.value.args[0] == "Cannot set slice for 2D image"

    def test_slice_disabled_for_no_data(self):
        client = ImageClient(self.collect, figure=FIGURE)
        assert client.slice_ind is None
        with pytest.raises(IndexError) as exc:
            client.slice_ind = 10
        assert exc.value.args[0] == "Cannot set slice for 2D image"

    def test_slice_enabled_for_3D(self):
        client = self.create_client_with_cube()
        assert client.slice_ind is not None
        client.slice_ind = 5
        assert client.slice_ind == 5

    def test_add_subset_via_method(self):
        client = ImageClient(self.collect, figure=FIGURE)
        self.collect.append(self.im)
        s = self.im.new_subset()
        client.add_layer(s)
        assert s in client.artists

    def test_remove_data(self):
        client = ImageClient(self.collect, figure=FIGURE)
        self.collect.append(self.im)
        s = self.im.new_subset()
        client.add_layer(self.im)
        assert self.im in client.artists
        assert s in client.artists
        client.delete_layer(self.im)
        assert client.display_data is None
        assert not self.im in client.artists
        assert not s in client.artists

    def test_set_norm(self):
        client = self.create_client_with_image()
        assert client.display_data is not None
        client.set_norm(vmin=10, vmax=100)
        for a in client.artists[self.im]:
            assert a.norm.vmin == 10
            assert a.norm.vmax == 100

    def test_delete_data(self):
        client = self.create_client_with_image()
        client.delete_layer(self.im)
        assert not self.im in client.artists

    def test_set_attribute(self):
        client = self.create_client_with_image()
        atts = self.im.component_ids()
        assert len(atts) > 1
        for att in atts:
            client.set_attribute(att)
            assert client.display_attribute is att

    def test_get_attribute(self):
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
        with pytest.raises(IndexError) as exc:
            client.slice_ind = 10
        assert exc.value.args[0] == "Cannot set slice for 2D image"

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
        with pytest.raises(IndexError) as exc:
            client.set_slice_ori(0)
        assert exc.value.args[0] == "Cannot set orientation of 2D image"

    def test_slice_ori_out_of_bounds(self):
        client = self.create_client_with_image()
        self.collect.append(self.cube)
        client.set_data(self.cube)
        with pytest.raises(TypeError) as exc:
            client.set_slice_ori(100)
        assert exc.value.args[0] == "Orientation must be 0, 1, or 2"

    def test_apply_roi_2d(self):
        """apply_roi is applied to all edit_subsets"""
        client = self.create_client_with_image()

        roi = core.roi.PolygonalROI(vx=[10, 20, 20, 10],
                                    vy=[10, 10, 20, 20])
        client.apply_roi(roi)
        roi2 = self.im.edit_subset.subset_state.roi
        state = self.im.edit_subset.subset_state

        assert roi2.to_polygon()[0] == roi.to_polygon()[0]
        assert roi2.to_polygon()[1] == roi.to_polygon()[1]
        assert state.xatt is self.im.get_pixel_component_id(1)
        assert state.yatt is self.im.get_pixel_component_id(0)

    def test_apply_roi_3d(self):
        client = self.create_client_with_cube()
        self.cube.coords = DummyCoords()
        roi = core.roi.PolygonalROI(vx=[10, 20, 20, 10],
                                    vy=[10, 10, 20, 20])

        client.set_slice_ori(0)
        client.apply_roi(roi)
        state = self.cube.edit_subset.subset_state
        roi2 = state.roi
        assert state.xatt is self.cube.get_pixel_component_id(2)
        assert state.yatt is self.cube.get_pixel_component_id(1)
        assert roi2.to_polygon()[0] == roi.to_polygon()[0]
        assert roi2.to_polygon()[1] == roi.to_polygon()[1]

        client.set_slice_ori(1)
        client.apply_roi(roi)
        state = self.cube.edit_subset.subset_state
        roi2 = state.roi
        assert state.xatt is self.cube.get_pixel_component_id(2)
        assert state.yatt is self.cube.get_pixel_component_id(0)
        assert roi2.to_polygon()[0] == roi.to_polygon()[0]
        assert roi2.to_polygon()[1] == roi.to_polygon()[1]

        client.set_slice_ori(2)
        client.apply_roi(roi)
        state = self.cube.edit_subset.subset_state
        roi2 = state.roi
        assert state.xatt is self.cube.get_pixel_component_id(1)
        assert state.yatt is self.cube.get_pixel_component_id(0)
        assert roi2.to_polygon()[0] == roi.to_polygon()[0]
        assert roi2.to_polygon()[1] == roi.to_polygon()[1]

    def test_update_subset_deletes_artist_on_error(self):
        client = self.create_client_with_image()
        sub = self.im.edit_subset

        bad_state = MagicMock()
        err = core.exceptions.IncompatibleAttribute("Can't make mask")
        bad_state.to_mask.side_effect = err
        bad_state.to_index_list.side_effect = err
        sub.subset_state = bad_state

        m = MagicMock()
        client.artists[sub][0].clear = m
        client._update_subset_single(sub)
        assert m.call_count == 1

    def test_subsets_shown_on_init(self):
        client = self.create_client_with_image()
        subset = self.im.edit_subset
        assert subset in client.artists

    def test_axis_labels(self):
        client = self.create_client_with_image()
        client.refresh()
        ax = client.axes
        assert ax.get_xlabel() == 'World 1'
        assert ax.get_ylabel() == 'World 0'

    def test_add_scatter_layer(self):
        client = self.create_client_with_image_and_scatter()
        assert self.scatter in client.artists
        for a in client.artists[self.scatter]:
            assert a.visible

    def test_check_update(self):
        client = self.create_client_with_image()
        mm = MagicMock()
        client._redraw = mm
        client.check_update(None)
        ct = mm.call_count
        client.check_update(None)
        assert mm.call_count == ct

        client.axes.set_xlim(100, 500)
        client.check_update(None)
        assert mm.call_count > ct

    def test_set_cmap(self):
        from matplotlib.cm import bone
        client = self.create_client_with_image()
        client.set_data(self.im)
        client.set_cmap(bone)
        for a in client.artists[self.im]:
            assert a.cmap is bone

    def test_bad_attribute(self):
        """Shoudl raise IncompatibleAttribute on bad input"""
        client = self.create_client_with_image()
        client.set_data(self.im)
        with pytest.raises(IncompatibleAttribute) as exc:
            client.set_attribute('bad')
        assert exc.value.args[0] == "Attribute not in data's attributes: bad"

    def test_sticky_norm(self):
        """Norm scaling for each component should be remembered"""
        client = self.create_client_with_image()
        x = self.im[self.im.visible_components[0]]
        y = x * 2
        self.im.add_component(y, 'y')

        client.set_attribute(self.im.visible_components[0])
        client.set_norm(1, 2)
        assert client.get_norm() == (1, 2)

        client.set_attribute(self.im.visible_components[1])
        client.set_norm(3, 4)

        client.set_attribute(self.im.visible_components[0])
        assert client.get_norm() == (1, 2)

    def test_scatter_persistent(self):
        """Ensure that updates to data plot don't erase scatter artists"""
        client = self.create_client_with_image_and_scatter()
        assert self.scatter in client.artists
        client._update_data_plot()
        assert self.scatter in client.artists

    def test_image_hide_persistent(self):
        """If image layer is disabled, it should stay disabled after update"""
        client = self.create_client_with_image()
        assert client.is_visible(self.im)
        client.set_visible(self.im, False)
        client.axes.set_xlim(1, 2)
        client.check_update(None)
        for a in client.artists[self.im]:
            for aa in a.artists:
                assert not aa.get_visible()


def test_format_coord_2d():
    """Coordinate display is in world coordinates"""

    d = core.Data(x=[[1, 2, 3], [2, 3, 4]])
    d.coords = DummyCoords()

    dc = core.DataCollection([d])
    c = ImageClient(dc)
    c.add_layer(d)
    ax = c.axes

    #no data set. Use default
    xy = ax.format_coord(1, 2)
    assert xy == 'x=1            y=2           '

    #use coord object
    c.set_data(d)
    xy = ax.format_coord(1, 2)
    assert xy == 'World 1=1          World 0=4'


def test_format_coord_3d():
    """Coordinate display is in world coordinates"""

    d = core.Data(x=[[[1, 2, 3], [2, 3, 4]], [[2, 3, 4], [3, 4, 5]]])
    d.coords = DummyCoords()

    dc = core.DataCollection([d])
    c = ImageClient(dc)
    c.add_layer(d)
    ax = c.axes

    #no data set. Use default
    xy = ax.format_coord(1, 2)
    assert xy == 'x=1            y=2           '

    #ori = 0
    c.set_data(d)
    c.set_slice_ori(0)  # constant z
    xy = ax.format_coord(1, 2)
    assert xy == 'World 2=1          World 1=4'

    c.set_slice_ori(1)  # constant y
    xy = ax.format_coord(1, 2)
    assert xy == 'World 2=1          World 0=6'

    c.set_slice_ori(2)  # constant x
    xy = ax.format_coord(1, 2)
    assert xy == 'World 1=2          World 0=6'
