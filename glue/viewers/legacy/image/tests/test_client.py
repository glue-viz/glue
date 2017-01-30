# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import pytest
import numpy as np
from mock import MagicMock

from glue.core.link_helpers import LinkSame
from glue.core.exceptions import IncompatibleAttribute
from glue import core
from glue.tests import example_data
from glue.utils import renderless_figure

from ..client import MplImageClient
from ..layer_artist import RGBImageLayerArtist, ImageLayerArtist


FIGURE = renderless_figure()


class DummyCoords(core.coordinates.Coordinates):

    def pixel2world(self, *args):
        return tuple(a * (i + 1) for i, a in enumerate(args))


class TrueState(core.subset.SubsetState):

    def to_mask(self, view=None):
        data = np.ones(self.parent.data.shape, dtype=bool)
        if view is not None:
            data = data[view]
        return data


class _TestImageClientBase(object):

    def setup_method(self, method):
        self.im = example_data.test_image()
        self.cube = example_data.test_cube()
        self.cube4 = core.Data(x=np.ones((2, 3, 4, 5)))
        self.scatter = core.Data(x=[1, 2, 3, 4], y=[4, 5, 6, 7], z=[0, 1, 2, 3])
        self.im.edit_subset = self.im.new_subset()
        self.cube.edit_subset = self.cube.new_subset()
        self.collect = core.data_collection.DataCollection()
        FIGURE.canvas.draw.reset_mock()

    def new_client(self, dc=None, figure=FIGURE):
        raise NotImplementedError()

    def create_client_with_image(self, **kwargs):
        client = self.new_client(**kwargs)
        self.collect.append(self.im)
        client.set_data(self.im)
        return client

    def create_client_with_hypercube(self):
        client = self.new_client()
        self.collect.append(self.cube4)
        client.set_data(self.cube4)
        return client

    def create_client_with_cube_and_scatter(self):

        client = self.create_client_with_cube()
        self.collect.append(self.cube)

        ix = self.cube.get_pixel_component_id(0)
        iy = self.cube.get_pixel_component_id(1)
        iz = self.cube.get_pixel_component_id(2)

        self.collect.add_link(LinkSame(self.scatter.id['x'], ix))
        self.collect.add_link(LinkSame(self.scatter.id['y'], iy))
        self.collect.add_link(LinkSame(self.scatter.id['z'], iz))
        client.add_scatter_layer(self.scatter)

        return client

    def create_client_with_image_and_scatter(self):

        client = self.create_client_with_image()
        self.collect.append(self.scatter)

        ix = self.im.get_world_component_id(0)
        iy = self.im.get_world_component_id(1)

        self.collect.add_link(LinkSame(self.scatter.id['x'], ix))
        self.collect.add_link(LinkSame(self.scatter.id['y'], iy))
        client.add_scatter_layer(self.scatter)

        return client

    def create_client_with_cube(self):
        client = self.new_client()
        self.collect.append(self.cube)
        client.set_data(self.cube)
        return client

    def test_empty_creation(self):
        client = self.new_client()
        assert client.display_data is None

    def test_nonempty_creation(self):
        self.collect.append(self.im)
        client = self.new_client()
        assert client.display_data is None
        assert not self.im in client.artists

    def test_invalid_add(self):
        client = self.new_client()
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
        assert exc.value.args[0] == "Can only set slice_ind for 3D images"

    def test_slice_disabled_for_no_data(self):
        client = self.new_client()
        assert client.slice_ind is None
        with pytest.raises(IndexError) as exc:
            client.slice_ind = 10
        assert exc.value.args[0] == "Can only set slice_ind for 3D images"

    def test_slice_enabled_for_3D(self):
        client = self.create_client_with_cube()
        assert client.slice_ind is not None
        client.slice_ind = 5
        assert client.slice_ind == 5

    def test_add_subset_via_method(self):
        client = self.new_client()
        self.collect.append(self.im)
        s = self.im.new_subset()
        client.add_layer(s)
        assert s in client.artists

    def test_remove_data(self):
        client = self.new_client()
        self.collect.append(self.im)
        s = self.im.new_subset()
        client.add_layer(self.im)
        assert self.im in client.artists
        assert s in client.artists
        client.delete_layer(self.im)
        assert client.display_data is not self.im
        assert not self.im in client.artists
        assert not s in client.artists

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

    def test_slice_ori_on_2d_raises(self):
        client = self.create_client_with_image()
        with pytest.raises(IndexError) as exc:
            client.set_slice_ori(0)
        assert exc.value.args[0] == "Can only set slice_ori for 3D images"

    def test_slice_ori_out_of_bounds(self):
        client = self.create_client_with_image()
        self.collect.append(self.cube)
        client.set_data(self.cube)
        with pytest.raises(ValueError) as exc:
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

    def test_subsets_shown_on_init(self):
        client = self.create_client_with_image()
        subset = self.im.edit_subset
        assert subset in client.artists

    def test_add_scatter_layer(self):
        client = self.create_client_with_image_and_scatter()
        assert self.scatter in client.artists
        for a in client.artists[self.scatter]:
            assert a.visible

    def test_data_scatter_emphasis_updates_on_slice_change(self):
        # regression test for 367
        client = self.create_client_with_cube_and_scatter()
        layer = client.artists[self.scatter][0]
        emph0 = layer.emphasis
        client.slice = (2, 'y', 'x')
        assert layer.emphasis is not emph0

    def test_scatter_persistent(self):
        """Ensure that updates to data plot don't erase scatter artists"""
        client = self.create_client_with_image_and_scatter()
        assert self.scatter in client.artists
        client._update_data_plot()
        assert self.scatter in client.artists

    def test_scatter_sync(self):
        """ Regression test for #360 """
        client = self.create_client_with_image_and_scatter()
        client.register_to_hub(self.collect.hub)
        self.scatter.label = 'scatter'

        sg = self.collect.new_subset_group()
        subset = sg.subsets[-1]
        assert subset.data is self.scatter

        client.add_scatter_layer(subset)
        art = client.artists[subset][0].artists

        sg.subset_state = self.scatter.id['x'] > 2
        client._update_subset_single(subset)
        assert client.artists[subset][0].artists is not art

    def test_scatter_subsets_not_auto_added(self):
        """Scatter subsets should not be added by
        SubsetAddMessage"""
        c = self.create_client_with_image()

        self.collect.append(self.scatter)

        c.register_to_hub(self.collect.hub)

        s = self.scatter.new_subset()
        assert s not in c.artists

    def test_scatter_layer_does_not_set_display_data(self):
        c = self.create_client_with_image()
        self.collect.append(self.scatter)
        d = c.display_data
        c.set_data(self.scatter)
        assert c.display_data is d

    def test_4d(self):
        c = self.create_client_with_hypercube()
        assert c.display_data is self.cube4

    def test_format_coord_works_without_data(self):
        # regression test for 402
        client = self.new_client()
        expected = dict(labels=['x=3', 'y=5'],
                        pix=(3, 5), world=(3, 5), value=np.nan)
        assert client.point_details(3, 5) == expected

    def test_visibility_toggles(self):

        c = self.create_client_with_image()
        s = self.im.edit_subset
        c.add_layer(s)

        c.set_visible(self.im, False)
        assert not c.is_visible(self.im)
        assert c.is_visible(s)

        c.set_visible(self.im, True)
        assert c.is_visible(self.im)
        assert c.is_visible(s)

        c.set_visible(s, False)
        assert c.is_visible(self.im)
        assert not c.is_visible(s)

    def test_component_replaced(self):
        # Regression test for #508
        c = self.create_client_with_image()

        d = c.display_data
        a = c.display_attribute
        test = core.ComponentID('test')

        c.register_to_hub(d.hub)
        d.update_id(a, test)

        assert c.display_attribute is test


class TestMplImageClient(_TestImageClientBase):

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

    def new_client(self, dc=None, figure=FIGURE):
        dc = dc or self.collect
        return MplImageClient(dc, figure=figure)

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

    def test_set_norm(self):
        client = self.create_client_with_image()
        assert client.display_data is not None
        client.set_norm(clip_lo=3, clip_hi=97)
        for a in client.artists[self.im]:
            assert a.norm.clip_lo == 3
            assert a.norm.clip_hi == 97

    def test_apply_roi_draws_once(self):
        assert MplImageClient.apply_roi._is_deferred

    def test_update_subset_deletes_artist_on_error(self):
        client = self.create_client_with_image()
        sub = self.im.edit_subset

        bad_state = MagicMock(spec_set=core.subset.SubsetState)
        err = core.exceptions.IncompatibleAttribute("Can't make mask")
        bad_state.to_mask.side_effect = err
        bad_state.to_index_list.side_effect = err
        sub.subset_state = bad_state

        m = MagicMock()
        client.artists[sub][0].clear = m
        client._update_subset_single(sub)
        assert m.call_count == 2

    def test_axis_labels(self):
        client = self.create_client_with_image()
        client.refresh()
        ax = client.axes
        assert ax.get_xlabel() == 'World 1'
        assert ax.get_ylabel() == 'World 0'

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

        client.set_attribute(self.
                             im.visible_components[0])
        client.set_norm(clip_lo=7, clip_hi=80)
        n = client.get_norm()
        assert n.clip_lo == 7
        assert n.clip_hi == 80

        client.set_attribute(self.im.visible_components[1])
        client.set_norm(clip_lo=20, clip_hi=30)

        client.set_attribute(self.im.visible_components[0])
        n == client.get_norm()
        assert n.clip_lo == 7
        assert n.clip_hi == 80

    def test_rgb_mode_toggle(self):
        c = self.create_client_with_image()
        im = c.rgb_mode(True)
        assert isinstance(im, RGBImageLayerArtist)
        assert c.rgb_mode() is im
        assert isinstance(c.rgb_mode(False), ImageLayerArtist)
        assert c.rgb_mode() is None

    def test_rgb_enabled_on_creation(self):
        """
        Artist show render when first created.
        Regression test for #419
        """
        c = self.create_client_with_image()
        artist = c.rgb_mode(True)
        assert artist.enabled

    def test_transpose(self):
        c = self.create_client_with_image()
        shp = self.im.shape
        c.slice = 'x', 'y'
        assert c.axes.get_xlim() == (0, shp[0])
        assert c.axes.get_ylim() == (0, shp[1])
        assert c.axes.get_xlabel() == 'World 0'
        assert c.axes.get_ylabel() == 'World 1'

    def test_slice_move_retains_zoom(self):
        # regression test for #224
        c = self.create_client_with_cube()
        c.axes.set_xlim(2, 11)
        c.axes.set_ylim(4, 11)
        c.slice = 1, 'y', 'x'
        assert c.axes.get_xlim() == (2, 11)
        assert c.axes.get_ylim() == (4, 11)


def test_format_coord_2d():
    """Coordinate display is in world coordinates"""

    d = core.Data(x=[[1, 2, 3], [2, 3, 4]])
    d.coords = DummyCoords()

    dc = core.DataCollection([d])
    c = MplImageClient(dc, figure=FIGURE)
    c.add_layer(d)
    ax = c.axes

    # no data set. Use default
    c.display_data = None
    xy = ax.format_coord(1, 2)
    assert xy == 'x=1            y=2           '

    # use coord object
    c.set_data(d)
    xy = ax.format_coord(1, 2)
    assert xy == 'World 0=4         World 1=1'


def test_format_coord_3d():
    """Coordinate display is in world coordinates"""

    d = core.Data(x=[[[1, 2, 3], [2, 3, 4]], [[2, 3, 4], [3, 4, 5]]])
    d.coords = DummyCoords()

    dc = core.DataCollection([d])
    c = MplImageClient(dc)
    c.add_layer(d)
    ax = c.axes

    # no data set. Use default
    c.display_data = None
    xy = ax.format_coord(1, 2)
    assert xy == 'x=1            y=2           '

    #ori = 0
    c.set_data(d)
    c.set_slice_ori(0)  # constant z
    xy = ax.format_coord(1, 2)
    assert xy == 'World 0=0         World 1=4         World 2=1'

    c.set_slice_ori(1)  # constant y
    xy = ax.format_coord(1, 2)
    assert xy == 'World 0=6         World 1=0         World 2=1'

    c.set_slice_ori(2)  # constant x
    xy = ax.format_coord(1, 2)
    assert xy == 'World 0=6         World 1=2         World 2=0'


class TestRGBImageLayerArtist(object):

    def setup_method(self, method):
        self.ax = MagicMock('matplotlib.axes.Axes')
        self.data = MagicMock('glue.core.Data')
        self.artist = RGBImageLayerArtist(self.data, self.ax)

    def test_set_norm(self):
        a = self.artist
        for c, n in zip(['red', 'green', 'blue'],
                        ['rnorm', 'gnorm', 'bnorm']):
            a.contrast_layer = c
            a.set_norm(vmin=5)
            assert getattr(a, n).vmin == 5
