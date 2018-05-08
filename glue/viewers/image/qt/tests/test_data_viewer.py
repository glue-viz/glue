# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import os
import gc
from collections import Counter

import pytest

from astropy.wcs import WCS

import numpy as np
from numpy.testing import assert_allclose

from glue.external.modest_image import ModestImage
from glue.core.coordinates import Coordinates, WCSCoordinates
from glue.core.message import SubsetUpdateMessage
from glue.core import HubListener, Data
from glue.core.roi import XRangeROI, RectangularROI
from glue.core.subset import RoiSubsetState
from glue.utils.qt import combo_as_string
from glue.viewers.matplotlib.qt.tests.test_data_viewer import BaseTestMatplotlibDataViewer
from glue.core.state import GlueUnSerializer
from glue.app.qt.layer_tree_widget import LayerTreeWidget
from glue.viewers.scatter.state import ScatterLayerState
from glue.viewers.image.state import ImageLayerState, ImageSubsetLayerState, AggregateSlice
from glue.core.link_helpers import LinkSame
from glue.app.qt import GlueApplication

from ..data_viewer import ImageViewer

DATA = os.path.join(os.path.dirname(__file__), 'data')


class TestImageCommon(BaseTestMatplotlibDataViewer):

    def init_data(self):
        return Data(label='d1', x=np.arange(24).reshape((2, 3, 4)), y=np.ones((2, 3, 4)))

    viewer_cls = ImageViewer

    @pytest.mark.skip()
    def test_double_add_ignored(self):
        pass

    def test_slice_change_single_draw(self):

        # Regression test for a bug that caused Matplotlib to draw once per
        # data/subset when changing slices.

        self.viewer.add_data(self.data)

        self.data_collection.new_subset_group(label='a', subset_state=self.data.id['x'] > 1)
        self.data_collection.new_subset_group(label='b', subset_state=self.data.id['x'] > 2)
        self.data_collection.new_subset_group(label='c', subset_state=self.data.id['x'] > 3)

        self.init_draw_count()

        assert self.draw_count == 0
        self.viewer.state.slices = (1, 1, 1)
        assert self.draw_count == 1


class MyCoords(Coordinates):
    def axis_label(self, i):
        return ['Banana', 'Apple'][i]


class TestImageViewer(object):

    def setup_method(self, method):

        self.coords = MyCoords()
        self.image1 = Data(label='image1', x=[[1, 2], [3, 4]], y=[[4, 5], [2, 3]])
        self.image2 = Data(label='image2', a=[[3, 3], [2, 2]], b=[[4, 4], [3, 2]],
                           coords=self.coords)
        self.catalog = Data(label='catalog', c=[1, 3, 2], d=[4, 3, 3])
        self.hypercube = Data(label='hypercube', x=np.arange(120).reshape((2, 3, 4, 5)))

        # Create data versions with WCS coordinates
        self.image1_wcs = Data(label='image1_wcs', x=self.image1['x'],
                               coords=WCSCoordinates(wcs=WCS(naxis=2)))
        self.hypercube_wcs = Data(label='hypercube_wcs', x=self.hypercube['x'],
                                  coords=WCSCoordinates(wcs=WCS(naxis=4)))

        self.application = GlueApplication()

        self.session = self.application.session

        self.hub = self.session.hub

        self.data_collection = self.session.data_collection
        self.data_collection.append(self.image1)
        self.data_collection.append(self.image2)
        self.data_collection.append(self.catalog)
        self.data_collection.append(self.hypercube)
        self.data_collection.append(self.image1_wcs)
        self.data_collection.append(self.hypercube_wcs)

        self.viewer = self.application.new_data_viewer(ImageViewer)

        self.data_collection.register_to_hub(self.hub)
        self.viewer.register_to_hub(self.hub)

        self.options_widget = self.viewer.options_widget()

    def teardown_method(self, method):
        self.viewer.close()
        self.viewer = None
        self.application.close()
        self.application = None

    def test_basic(self):

        # Check defaults when we add data

        self.viewer.add_data(self.image1)

        assert combo_as_string(self.options_widget.ui.combosel_x_att_world) == 'Coordinate components:World 0:World 1'
        assert combo_as_string(self.options_widget.ui.combosel_y_att_world) == 'Coordinate components:World 0:World 1'

        assert self.viewer.axes.get_xlabel() == 'World 1'
        assert self.viewer.state.x_att_world is self.image1.id['World 1']
        assert self.viewer.state.x_att is self.image1.pixel_component_ids[1]
        # TODO: make sure limits are deterministic then update this
        # assert self.viewer.state.x_min == -0.5
        # assert self.viewer.state.x_max == +1.5

        assert self.viewer.axes.get_ylabel() == 'World 0'
        assert self.viewer.state.y_att_world is self.image1.id['World 0']
        assert self.viewer.state.y_att is self.image1.pixel_component_ids[0]
        # TODO: make sure limits are deterministic then update this
        # assert self.viewer.state.y_min == -0.5
        # assert self.viewer.state.y_max == +1.5

        assert not self.viewer.state.x_log
        assert not self.viewer.state.y_log

        assert len(self.viewer.state.layers) == 1

    def test_custom_coords(self):

        # Check defaults when we add data with coordinates

        self.viewer.add_data(self.image2)

        assert combo_as_string(self.options_widget.ui.combosel_x_att_world) == 'Coordinate components:Banana:Apple'
        assert combo_as_string(self.options_widget.ui.combosel_x_att_world) == 'Coordinate components:Banana:Apple'

        assert self.viewer.axes.get_xlabel() == 'Apple'
        assert self.viewer.state.x_att_world is self.image2.id['Apple']
        assert self.viewer.state.x_att is self.image2.pixel_component_ids[1]
        assert self.viewer.axes.get_ylabel() == 'Banana'
        assert self.viewer.state.y_att_world is self.image2.id['Banana']
        assert self.viewer.state.y_att is self.image2.pixel_component_ids[0]

    def test_flip(self):

        self.viewer.add_data(self.image1)

        x_min_start = self.viewer.state.x_min
        x_max_start = self.viewer.state.x_max

        self.options_widget.button_flip_x.click()

        assert self.viewer.state.x_min == x_max_start
        assert self.viewer.state.x_max == x_min_start

        y_min_start = self.viewer.state.y_min
        y_max_start = self.viewer.state.y_max

        self.options_widget.button_flip_y.click()

        assert self.viewer.state.y_min == y_max_start
        assert self.viewer.state.y_max == y_min_start

    def test_combo_updates_with_component_add(self):
        self.viewer.add_data(self.image1)
        self.image1.add_component([[9, 9], [8, 8]], 'z')
        assert self.viewer.state.x_att_world is self.image1.id['World 1']
        assert self.viewer.state.y_att_world is self.image1.id['World 0']
        # TODO: there should be an easier way to do this
        layer_style_editor = self.viewer._view.layout_style_widgets[self.viewer.layers[0]]
        assert combo_as_string(layer_style_editor.ui.combosel_attribute) == 'x:y:z'

    def test_apply_roi(self):

        self.viewer.add_data(self.image1)

        roi = RectangularROI(0.4, 1.6, -0.6, 0.6)

        assert len(self.viewer.layers) == 1

        self.viewer.apply_roi(roi)

        assert len(self.viewer.layers) == 2
        assert len(self.image1.subsets) == 1

        assert_allclose(self.image1.subsets[0].to_mask(), [[0, 1], [0, 0]])

        state = self.image1.subsets[0].subset_state
        assert isinstance(state, RoiSubsetState)

    def test_apply_roi_empty(self):
        # Make sure that doing an ROI selection on an empty viewer doesn't
        # produce error messsages
        roi = XRangeROI(-0.2, 0.1)
        self.viewer.apply_roi(roi)

    def test_identical(self):

        # Check what happens if we set both attributes to the same coordinates

        self.viewer.add_data(self.image2)

        assert self.viewer.state.x_att_world is self.image2.id['Apple']
        assert self.viewer.state.y_att_world is self.image2.id['Banana']

        self.viewer.state.y_att_world = self.image2.id['Apple']

        assert self.viewer.state.x_att_world is self.image2.id['Banana']
        assert self.viewer.state.y_att_world is self.image2.id['Apple']

        self.viewer.state.x_att_world = self.image2.id['Apple']

        assert self.viewer.state.x_att_world is self.image2.id['Apple']
        assert self.viewer.state.y_att_world is self.image2.id['Banana']

    def test_duplicate_subsets(self):

        # Regression test: make sure that when adding a seconda layer for the
        # same dataset, we don't add the subsets all over again.

        self.viewer.add_data(self.image1)
        self.data_collection.new_subset_group(subset_state=self.image1.id['x'] > 1, label='A')

        assert len(self.viewer.layers) == 2

        self.viewer.add_data(self.image1)

        assert len(self.viewer.layers) == 3

    def test_aspect_subset(self):

        self.viewer.add_data(self.image1)

        assert self.viewer.state.aspect == 'equal'
        assert self.viewer.axes.get_aspect() == 'equal'

        self.viewer.state.aspect = 'auto'

        self.data_collection.new_subset_group('s1', self.image1.id['x'] > 0.)

        assert len(self.viewer.state.layers) == 2

        assert self.viewer.state.aspect == 'auto'
        assert self.viewer.axes.get_aspect() == 'auto'

        self.viewer.state.aspect = 'equal'

        self.data_collection.new_subset_group('s2', self.image1.id['x'] > 1.)

        assert len(self.viewer.state.layers) == 3

        assert self.viewer.state.aspect == 'equal'
        assert self.viewer.axes.get_aspect() == 'equal'

    def test_hypercube(self):

        # Check defaults when we add data

        self.viewer.add_data(self.hypercube)

        assert combo_as_string(self.options_widget.ui.combosel_x_att_world) == 'Coordinate components:World 0:World 1:World 2:World 3'
        assert combo_as_string(self.options_widget.ui.combosel_x_att_world) == 'Coordinate components:World 0:World 1:World 2:World 3'

        assert self.viewer.axes.get_xlabel() == 'World 3'
        assert self.viewer.state.x_att_world is self.hypercube.id['World 3']
        assert self.viewer.state.x_att is self.hypercube.pixel_component_ids[3]
        # TODO: make sure limits are deterministic then update this
        # assert self.viewer.state.x_min == -0.5
        # assert self.viewer.state.x_max == +1.5

        assert self.viewer.axes.get_ylabel() == 'World 2'
        assert self.viewer.state.y_att_world is self.hypercube.id['World 2']
        assert self.viewer.state.y_att is self.hypercube.pixel_component_ids[2]
        # TODO: make sure limits are deterministic then update this
        # assert self.viewer.state.y_min == -0.5
        # assert self.viewer.state.y_max == +1.5

        assert not self.viewer.state.x_log
        assert not self.viewer.state.y_log

        assert len(self.viewer.state.layers) == 1

    def test_hypercube_world(self):

        # Check defaults when we add data

        wcs = WCS(naxis=4)
        hypercube2 = Data()
        hypercube2.coords = WCSCoordinates(wcs=wcs)
        hypercube2.add_component(np.random.random((2, 3, 4, 5)), 'a')

        self.data_collection.append(hypercube2)

        self.viewer.add_data(hypercube2)

    def test_incompatible_subset(self):
        self.viewer.add_data(self.image1)
        self.data_collection.new_subset_group(subset_state=self.catalog.id['c'] > 1, label='A')

    def test_invisible_subset(self):

        # Regression test for a bug that caused a subset layer that started
        # off as invisible to have issues when made visible. We emulate the
        # initial invisible (but enabled) state by invalidating the cache.

        self.viewer.add_data(self.image1)
        self.data_collection.new_subset_group(subset_state=self.image1.id['x'] > 1, label='A')
        self.viewer.layers[1].visible = False
        self.viewer.layers[1].image_artist.invalidate_cache()
        self.viewer.layers[1].redraw()
        assert not np.any(self.viewer.layers[1].image_artist._A.mask)
        self.viewer.layers[1].visible = True
        assert not np.any(self.viewer.layers[1].image_artist._A.mask)

    def test_apply_roi_single(self):

        # Regression test for a bug that caused mode.update to be called
        # multiple times and resulted in all other viewers receiving many
        # messages regarding subset updates (this occurred when multiple)
        # datasets were present.

        layer_tree = LayerTreeWidget(session=self.session)
        layer_tree.set_checkable(False)
        layer_tree.setup(self.data_collection)
        layer_tree.bind_selection_to_edit_subset()

        class Client(HubListener):

            def __init__(self, *args, **kwargs):
                super(Client, self).__init__(*args, **kwargs)
                self.count = Counter()

            def ping(self, message):
                self.count[message.sender] += 1

            def register_to_hub(self, hub):
                hub.subscribe(self, SubsetUpdateMessage, handler=self.ping)

        d1 = Data(a=[[1, 2], [3, 4]], label='d1')
        d2 = Data(b=[[1, 2], [3, 4]], label='d2')
        d3 = Data(c=[[1, 2], [3, 4]], label='d3')
        d4 = Data(d=[[1, 2], [3, 4]], label='d4')

        self.data_collection.append(d1)
        self.data_collection.append(d2)
        self.data_collection.append(d3)
        self.data_collection.append(d4)

        client = Client()
        client.register_to_hub(self.hub)

        self.viewer.add_data(d1)
        self.viewer.add_data(d3)

        roi = XRangeROI(2.5, 3.5)
        self.viewer.apply_roi(roi)

        for subset in client.count:
            assert client.count[subset] == 1

    def test_disable_incompatible(self):

        # Test to make sure that image and image subset layers are disabled if
        # their pixel coordinates are not compatible with the ones of the
        # reference data.

        self.viewer.add_data(self.image1)
        self.viewer.add_data(self.image2)

        assert self.viewer.state.reference_data is self.image1

        self.data_collection.new_subset_group()

        assert len(self.viewer.layers) == 4

        # Only the two layers associated with the reference data should be enabled
        for layer_artist in self.viewer.layers:
            if layer_artist.layer in (self.image1, self.image1.subsets[0]):
                assert layer_artist.enabled
            else:
                assert not layer_artist.enabled

        py1, px1 = self.image1.pixel_component_ids
        py2, px2 = self.image2.pixel_component_ids

        link1 = LinkSame(px1, px2)
        self.data_collection.add_link(link1)

        # One link isn't enough, second dataset layers are still not enabled

        for layer_artist in self.viewer.layers:
            if layer_artist.layer in (self.image1, self.image1.subsets[0]):
                assert layer_artist.enabled
            else:
                assert not layer_artist.enabled

        link2 = LinkSame(py1, py2)
        self.data_collection.add_link(link2)

        # All layers should now be enabled

        for layer_artist in self.viewer.layers:
            assert layer_artist.enabled

        self.data_collection.remove_link(link2)

        # We should now be back to the original situation

        for layer_artist in self.viewer.layers:
            if layer_artist.layer in (self.image1, self.image1.subsets[0]):
                assert layer_artist.enabled
            else:
                assert not layer_artist.enabled

    def test_change_reference_data(self, capsys):

        # Test to make sure everything works fine if we change the reference data.

        self.viewer.add_data(self.image1)
        self.viewer.add_data(self.image2)

        assert self.viewer.state.reference_data is self.image1
        assert self.viewer.state.x_att_world is self.image1.world_component_ids[-1]
        assert self.viewer.state.y_att_world is self.image1.world_component_ids[-2]
        assert self.viewer.state.x_att is self.image1.pixel_component_ids[-1]
        assert self.viewer.state.y_att is self.image1.pixel_component_ids[-2]

        self.viewer.state.reference_data = self.image2

        assert self.viewer.state.reference_data is self.image2
        assert self.viewer.state.x_att_world is self.image2.world_component_ids[-1]
        assert self.viewer.state.y_att_world is self.image2.world_component_ids[-2]
        assert self.viewer.state.x_att is self.image2.pixel_component_ids[-1]
        assert self.viewer.state.y_att is self.image2.pixel_component_ids[-2]

        self.viewer.state.reference_data = self.image1

        assert self.viewer.state.reference_data is self.image1
        assert self.viewer.state.x_att_world is self.image1.world_component_ids[-1]
        assert self.viewer.state.y_att_world is self.image1.world_component_ids[-2]
        assert self.viewer.state.x_att is self.image1.pixel_component_ids[-1]
        assert self.viewer.state.y_att is self.image1.pixel_component_ids[-2]

        # Some exceptions used to happen during callbacks, and these show up
        # in stderr but don't interrupt the code, so we make sure here that
        # nothing was printed to stdout nor stderr.

        out, err = capsys.readouterr()

        assert out.strip() == ""
        assert err.strip() == ""

    @pytest.mark.parametrize('wcs', [False, True])
    def test_change_reference_data_dimensionality(self, capsys, wcs):

        # Regression test for a bug that caused an exception when changing
        # the dimensionality of the reference data

        if wcs:
            first = self.image1_wcs
            second = self.hypercube_wcs
        else:
            first = self.image1
            second = self.hypercube

        self.viewer.add_data(first)
        self.viewer.add_data(second)

        assert self.viewer.state.reference_data is first
        assert self.viewer.state.x_att_world is first.world_component_ids[-1]
        assert self.viewer.state.y_att_world is first.world_component_ids[-2]
        assert self.viewer.state.x_att is first.pixel_component_ids[-1]
        assert self.viewer.state.y_att is first.pixel_component_ids[-2]

        self.viewer.state.reference_data = second

        assert self.viewer.state.reference_data is second
        assert self.viewer.state.x_att_world is second.world_component_ids[-1]
        assert self.viewer.state.y_att_world is second.world_component_ids[-2]
        assert self.viewer.state.x_att is second.pixel_component_ids[-1]
        assert self.viewer.state.y_att is second.pixel_component_ids[-2]

        self.viewer.state.reference_data = first

        assert self.viewer.state.reference_data is first
        assert self.viewer.state.x_att_world is first.world_component_ids[-1]
        assert self.viewer.state.y_att_world is first.world_component_ids[-2]
        assert self.viewer.state.x_att is first.pixel_component_ids[-1]
        assert self.viewer.state.y_att is first.pixel_component_ids[-2]

        # Some exceptions used to happen during callbacks, and these show up
        # in stderr but don't interrupt the code, so we make sure here that
        # nothing was printed to stdout nor stderr.

        out, err = capsys.readouterr()

        assert out.strip() == ""
        assert err.strip() == ""

    def test_scatter_overlay(self):
        self.viewer.add_data(self.image1)
        self.viewer.add_data(self.catalog)

    def test_removed_subset(self):

        # Regression test for a bug in v0.11.0 that meant that if a subset
        # was removed, the image viewer would then crash when changing view
        # (e.g. zooming in). The bug was caused by undeleted references to
        # ModestImage due to circular references. We therefore check in this
        # test how many ModestImage objects exist.

        def get_modest_images():
            mi = []
            gc.collect()
            for obj in gc.get_objects():
                try:
                    if isinstance(obj, ModestImage):
                        mi.append(obj)
                except ReferenceError:
                    pass
            return mi

        # The viewer starts off with one ModestImage. This is also a good test
        # that other ModestImages in other tests have been removed.
        assert len(get_modest_images()) == 1

        large_image = Data(x=np.random.random((2048, 2048)))
        self.data_collection.append(large_image)

        # The subset group can be made from any dataset
        subset_group = self.data_collection.new_subset_group(subset_state=self.image1.id['x'] > 1, label='A')

        self.viewer.add_data(large_image)

        # Since the dataset added has a subset, and each subset has its own
        # ModestImage, this increases the count.
        assert len(get_modest_images()) == 2

        assert len(self.viewer.layers) == 2

        self.data_collection.remove_subset_group(subset_group)

        # Removing the subset should bring the count back to 1 again
        assert len(get_modest_images()) == 1

    def test_select_previously_incompatible_layer(self):

        # Regression test for a bug that caused a selection in a previously disabled
        # layer to enable the layer without updating the subset view

        self.viewer.add_data(self.image1)
        self.viewer.add_data(self.catalog)
        self.catalog.add_component([4, 5, 6], 'e')

        link1 = LinkSame(self.catalog.id['c'], self.image1.world_component_ids[0])
        link2 = LinkSame(self.catalog.id['d'], self.image1.world_component_ids[1])
        self.data_collection.add_link(link1)
        self.data_collection.add_link(link2)

        self.data_collection.new_subset_group(subset_state=self.catalog.id['e'] > 4)

        assert self.viewer.layers[0].enabled  # image
        assert self.viewer.layers[1].enabled  # scatter
        assert not self.viewer.layers[2].enabled  # image subset
        assert self.viewer.layers[3].enabled  # scatter subset

        assert not self.viewer.layers[2].image_artist.get_visible()

        self.data_collection.subset_groups[0].subset_state = self.catalog.id['c'] > -1

        assert self.viewer.layers[0].enabled  # image
        assert self.viewer.layers[1].enabled  # scatter
        assert self.viewer.layers[2].enabled  # image subset
        assert self.viewer.layers[3].enabled  # scatter subset

        assert self.viewer.layers[2].image_artist.get_visible()

    def test_linking_and_enabling(self):

        # Regression test for a bug that caused layers not not be correctly
        # enabled/disabled.

        self.viewer.add_data(self.image1)
        self.viewer.add_data(self.catalog)
        self.catalog.add_component([4, 5, 6], 'e')

        self.data_collection.new_subset_group(subset_state=self.catalog.id['e'] > 4)

        assert self.viewer.layers[0].enabled  # image
        assert not self.viewer.layers[1].enabled  # scatter
        assert not self.viewer.layers[2].enabled  # image subset
        assert not self.viewer.layers[3].enabled  # scatter subset

        link1 = LinkSame(self.catalog.id['c'], self.image1.world_component_ids[0])
        link2 = LinkSame(self.catalog.id['d'], self.image1.world_component_ids[1])
        self.data_collection.add_link(link1)
        self.data_collection.add_link(link2)

        assert self.viewer.layers[0].enabled  # image
        assert self.viewer.layers[1].enabled  # scatter
        assert not self.viewer.layers[2].enabled  # image subset
        assert self.viewer.layers[3].enabled  # scatter subset

    def test_save_aggregate_slice(self, tmpdir):

        # Regression test to make sure that image viewers that include
        # aggregate slice objects in the slices can be saved/restored

        self.viewer.add_data(self.hypercube)
        self.viewer.state.slices = AggregateSlice(slice(1, 3), 10, np.sum), 3, 0, 0

        filename = tmpdir.join('session.glu').strpath

        self.application.save_session(filename)
        self.application.close()

        app2 = GlueApplication.restore_session(filename)
        viewer_state = app2.viewers[0][0].state
        slices = viewer_state.slices
        assert isinstance(slices[0], AggregateSlice)
        assert slices[0].slice == slice(1, 3)
        assert slices[0].center == 10
        assert slices[0].function is np.sum
        assert slices[1:] == (3, 0, 0)

        app2.close()

    def test_subset_cube_image(self):

        # Regression test to make sure that if an image and cube are present
        # in an image viewer and a subset is also present, we don't get an
        # error when trying to access the subset shape

        self.viewer.add_data(self.image1)
        self.data_collection.new_subset_group(label='subset',
                                              subset_state=self.image1.id['x'] > 1.5)

        self.viewer.add_data(self.hypercube)
        self.viewer.state.reference_data = self.hypercube

        assert self.viewer.layers[1].subset_array.shape == (4, 5)
        assert self.viewer.layers[3].subset_array.shape == (4, 5)

    def test_preserve_slice(self):

        # Regression test to make sure that when adding a second dataset to
        # an image viewer, the current slice in a cube does not change.

        self.viewer.add_data(self.hypercube)
        self.viewer.state.slices = (1, 2, 3, 4)
        self.viewer.add_data(self.image1)
        assert self.viewer.state.slices == (1, 2, 3, 4)

    def test_close(self):

        # Regression test for a bug that caused an error related to the toolbar
        # and _mpl_nav not being present when closing the viewer.

        self.viewer.toolbar.active_tool = self.viewer.toolbar.tools['mpl:zoom']
        self.viewer.close(warn=False)


class TestSessions(object):

    @pytest.mark.parametrize('protocol', [0, 1])
    def test_session_back_compat(self, protocol):

        filename = os.path.join(DATA, 'image_v{0}.glu'.format(protocol))

        with open(filename, 'r') as f:
            session = f.read()

        state = GlueUnSerializer.loads(session)

        ga = state.object('__main__')

        dc = ga.session.data_collection

        assert len(dc) == 2

        assert dc[0].label == 'data1'
        assert dc[1].label == 'data2'

        viewer1 = ga.viewers[0][0]

        assert len(viewer1.state.layers) == 3

        assert viewer1.state.x_att_world is dc[0].id['World 1']
        assert viewer1.state.y_att_world is dc[0].id['World 0']

        assert viewer1.state.x_min < -0.5
        assert viewer1.state.x_max > 1.5
        assert viewer1.state.y_min <= -0.5
        assert viewer1.state.y_max >= 1.5

        layer_state = viewer1.state.layers[0]
        assert isinstance(layer_state, ImageLayerState)
        assert layer_state.visible
        assert layer_state.bias == 0.5
        assert layer_state.contrast == 1.0
        assert layer_state.stretch == 'sqrt'
        assert layer_state.percentile == 99

        layer_state = viewer1.state.layers[1]
        assert isinstance(layer_state, ScatterLayerState)
        assert layer_state.visible

        layer_state = viewer1.state.layers[2]
        assert isinstance(layer_state, ImageSubsetLayerState)
        assert not layer_state.visible

        viewer2 = ga.viewers[0][1]

        assert len(viewer2.state.layers) == 2

        assert viewer2.state.x_att_world is dc[0].id['World 1']
        assert viewer2.state.y_att_world is dc[0].id['World 0']

        assert viewer2.state.x_min < -0.5
        assert viewer2.state.x_max > 1.5
        assert viewer2.state.y_min <= -0.5
        assert viewer2.state.y_max >= 1.5

        layer_state = viewer2.state.layers[0]
        assert layer_state.visible
        assert layer_state.stretch == 'arcsinh'
        assert layer_state.v_min == 1
        assert layer_state.v_max == 4

        layer_state = viewer2.state.layers[1]
        assert layer_state.visible

        viewer3 = ga.viewers[0][2]

        assert len(viewer3.state.layers) == 2

        assert viewer3.state.x_att_world is dc[0].id['World 1']
        assert viewer3.state.y_att_world is dc[0].id['World 0']

        assert viewer3.state.x_min < -0.5
        assert viewer3.state.x_max > 1.5
        assert viewer3.state.y_min <= -0.5
        assert viewer3.state.y_max >= 1.5

        layer_state = viewer3.state.layers[0]
        assert layer_state.visible
        assert layer_state.stretch == 'linear'
        assert layer_state.v_min == -2
        assert layer_state.v_max == 2

        layer_state = viewer3.state.layers[1]
        assert layer_state.visible

        ga.close()

    @pytest.mark.parametrize('protocol', [0, 1])
    def test_session_cube_back_compat(self, protocol):

        filename = os.path.join(DATA, 'image_cube_v{0}.glu'.format(protocol))

        with open(filename, 'r') as f:
            session = f.read()

        state = GlueUnSerializer.loads(session)

        ga = state.object('__main__')

        dc = ga.session.data_collection

        assert len(dc) == 1

        assert dc[0].label == 'array'

        viewer1 = ga.viewers[0][0]

        assert len(viewer1.state.layers) == 1

        assert viewer1.state.x_att_world is dc[0].id['World 2']
        assert viewer1.state.y_att_world is dc[0].id['World 1']
        assert viewer1.state.slices == [2, 0, 0, 1]

        ga.close()

    @pytest.mark.parametrize('protocol', [0, 1])
    def test_session_rgb_back_compat(self, protocol):

        filename = os.path.join(DATA, 'image_rgb_v{0}.glu'.format(protocol))

        with open(filename, 'r') as f:
            session = f.read()

        state = GlueUnSerializer.loads(session)

        ga = state.object('__main__')

        dc = ga.session.data_collection

        assert len(dc) == 1

        assert dc[0].label == 'rgbcube'

        viewer1 = ga.viewers[0][0]

        assert len(viewer1.state.layers) == 3
        assert viewer1.state.color_mode == 'One color per layer'

        layer_state = viewer1.state.layers[0]
        assert layer_state.visible
        assert layer_state.attribute.label == 'a'
        assert layer_state.color == 'r'

        layer_state = viewer1.state.layers[1]
        assert not layer_state.visible
        assert layer_state.attribute.label == 'c'
        assert layer_state.color == 'g'

        layer_state = viewer1.state.layers[2]
        assert layer_state.visible
        assert layer_state.attribute.label == 'b'
        assert layer_state.color == 'b'

        ga.close()
