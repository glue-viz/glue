# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import os
from collections import Counter

import pytest
import numpy as np

from numpy.testing import assert_allclose, assert_equal

from glue.config import colormaps
from glue.core.message import SubsetUpdateMessage
from glue.core import HubListener, Data
from glue.core.roi import XRangeROI, RectangularROI, CircularROI
from glue.core.subset import RoiSubsetState, AndState
from glue import core
from glue.core.component_id import ComponentID
from glue.utils.qt import combo_as_string
from glue.viewers.matplotlib.qt.tests.test_data_viewer import BaseTestMatplotlibDataViewer
from glue.core.state import GlueUnSerializer
from glue.app.qt.layer_tree_widget import LayerTreeWidget
from glue.app.qt import GlueApplication
from glue.tests.helpers import PYSIDE2_INSTALLED  # noqa

from ..data_viewer import ScatterViewer

DATA = os.path.join(os.path.dirname(__file__), 'data')


class TestScatterCommon(BaseTestMatplotlibDataViewer):
    def init_data(self):
        return Data(label='d1', x=[3.4, 2.3, -1.1, 0.3], y=['a', 'b', 'c', 'a'])
    viewer_cls = ScatterViewer


class TestScatterViewer(object):

    def setup_method(self, method):

        self.data = Data(label='d1', x=[3.4, 2.3, -1.1, 0.3],
                         y=[3.2, 3.3, 3.4, 3.5], z=['a', 'b', 'c', 'a'])
        self.data_2d = Data(label='d2', a=[[1, 2], [3, 4]], b=[[5, 6], [7, 8]],
                            x=[[3, 5], [5.4, 1]], y=[[1.2, 4], [7, 8]])

        self.app = GlueApplication()
        self.session = self.app.session
        self.hub = self.session.hub

        self.data_collection = self.session.data_collection
        self.data_collection.append(self.data)
        self.data_collection.append(self.data_2d)

        self.viewer = self.app.new_data_viewer(ScatterViewer)

    def teardown_method(self, method):
        self.viewer.close()
        self.viewer = None
        self.app.close()
        self.app = None

    def test_basic(self):

        viewer_state = self.viewer.state

        # Check defaults when we add data
        self.viewer.add_data(self.data)

        assert combo_as_string(self.viewer.options_widget().ui.combosel_x_att) == 'Main components:x:y:z:Coordinate components:Pixel Axis 0 [x]:World 0'
        assert combo_as_string(self.viewer.options_widget().ui.combosel_y_att) == 'Main components:x:y:z:Coordinate components:Pixel Axis 0 [x]:World 0'

        assert viewer_state.x_att is self.data.id['x']
        assert_allclose(viewer_state.x_min, -1.1 - 0.225)
        assert_allclose(viewer_state.x_max, 3.4 + 0.225)

        assert viewer_state.y_att is self.data.id['y']
        assert_allclose(viewer_state.y_min, 3.2 - 0.015)
        assert_allclose(viewer_state.y_max, 3.5 + 0.015)

        assert not viewer_state.x_log
        assert not viewer_state.y_log

        assert len(viewer_state.layers) == 1

        # Change to categorical component and check new values

        viewer_state.y_att = self.data.id['z']

        assert viewer_state.x_att is self.data.id['x']
        assert_allclose(viewer_state.x_min, -1.1 - 0.225)
        assert_allclose(viewer_state.x_max, 3.4 + 0.225)

        assert viewer_state.y_att is self.data.id['z']
        assert_allclose(viewer_state.y_min, -0.5 - 0.15)
        assert_allclose(viewer_state.y_max, 2.5 + 0.15)

        assert not viewer_state.x_log
        assert not viewer_state.y_log

    def test_flip(self):

        viewer_state = self.viewer.state

        self.viewer.add_data(self.data)

        assert_allclose(viewer_state.x_min, -1.1 - 0.225)
        assert_allclose(viewer_state.x_max, 3.4 + 0.225)

        self.viewer.options_widget().button_flip_x.click()

        assert_allclose(viewer_state.x_max, -1.1 - 0.225)
        assert_allclose(viewer_state.x_min, 3.4 + 0.225)

        assert_allclose(viewer_state.y_min, 3.2 - 0.015)
        assert_allclose(viewer_state.y_max, 3.5 + 0.015)

        self.viewer.options_widget().button_flip_y.click()

        assert_allclose(viewer_state.y_max, 3.2 - 0.015)
        assert_allclose(viewer_state.y_min, 3.5 + 0.015)

    def test_remove_data(self):
        self.viewer.add_data(self.data)
        assert combo_as_string(self.viewer.options_widget().ui.combosel_x_att) == 'Main components:x:y:z:Coordinate components:Pixel Axis 0 [x]:World 0'
        assert combo_as_string(self.viewer.options_widget().ui.combosel_y_att) == 'Main components:x:y:z:Coordinate components:Pixel Axis 0 [x]:World 0'
        self.data_collection.remove(self.data)
        assert combo_as_string(self.viewer.options_widget().ui.combosel_x_att) == ''
        assert combo_as_string(self.viewer.options_widget().ui.combosel_y_att) == ''

    def test_update_component_updates_title(self):
        self.viewer.add_data(self.data)
        assert self.viewer.windowTitle() == '2D Scatter'
        self.viewer.state.x_att = self.data.id['y']
        assert self.viewer.windowTitle() == '2D Scatter'

    def test_combo_updates_with_component_add(self):
        self.viewer.add_data(self.data)
        self.data.add_component([3, 4, 1, 2], 'a')
        assert self.viewer.state.x_att is self.data.id['x']
        assert self.viewer.state.y_att is self.data.id['y']
        assert combo_as_string(self.viewer.options_widget().ui.combosel_x_att) == 'Main components:x:y:z:a:Coordinate components:Pixel Axis 0 [x]:World 0'
        assert combo_as_string(self.viewer.options_widget().ui.combosel_y_att) == 'Main components:x:y:z:a:Coordinate components:Pixel Axis 0 [x]:World 0'

    def test_nonnumeric_first_component(self):
        # regression test for #208. Shouldn't complain if
        # first component is non-numerical
        data = core.Data()
        data.add_component(['a', 'b', 'c'], label='c1')
        data.add_component([1, 2, 3], label='c2')
        self.data_collection.append(data)
        self.viewer.add_data(data)

    def test_apply_roi(self):

        self.viewer.add_data(self.data)

        roi = RectangularROI(0, 3, 3.25, 3.45)

        assert len(self.viewer.layers) == 1

        self.viewer.apply_roi(roi)

        assert len(self.viewer.layers) == 2
        assert len(self.data.subsets) == 1

        assert_allclose(self.data.subsets[0].to_mask(), [0, 1, 0, 0])

        state = self.data.subsets[0].subset_state
        assert isinstance(state, RoiSubsetState)

    def test_apply_roi_categorical(self):

        viewer_state = self.viewer.state

        self.viewer.add_data(self.data)

        viewer_state.y_att = self.data.id['z']

        roi = RectangularROI(0, 3, -0.4, 0.3)

        assert len(self.viewer.layers) == 1

        self.viewer.apply_roi(roi)

        assert len(self.viewer.layers) == 2
        assert len(self.data.subsets) == 1

        assert_allclose(self.data.subsets[0].to_mask(), [0, 0, 0, 1])

        state = self.data.subsets[0].subset_state
        assert isinstance(state, AndState)

    def test_apply_roi_empty(self):
        # Make sure that doing an ROI selection on an empty viewer doesn't
        # produce error messsages
        roi = XRangeROI(-0.2, 0.1)
        self.viewer.apply_roi(roi)

    def test_axes_labels(self):

        viewer_state = self.viewer.state

        self.viewer.add_data(self.data)

        assert self.viewer.axes.get_xlabel() == 'x'
        assert self.viewer.axes.get_ylabel() == 'y'

        viewer_state.x_log = True

        assert self.viewer.axes.get_xlabel() == 'Log x'
        assert self.viewer.axes.get_ylabel() == 'y'

        viewer_state.x_att = self.data.id['y']

        assert self.viewer.axes.get_xlabel() == 'y'
        assert self.viewer.axes.get_ylabel() == 'y'

        viewer_state.y_log = True

        assert self.viewer.axes.get_xlabel() == 'y'
        assert self.viewer.axes.get_ylabel() == 'Log y'

    def test_component_replaced(self):

        # regression test for 508 - if a component ID is replaced, we should
        # make sure that the component ID is selected if the old component ID
        # was selected

        self.viewer.add_data(self.data)
        self.viewer.state.x_att = self.data.id['x']
        test = ComponentID('test')
        self.data.update_id(self.viewer.state.x_att, test)
        assert self.viewer.state.x_att is test
        assert combo_as_string(self.viewer.options_widget().ui.combosel_x_att) == 'Main components:test:y:z:Coordinate components:Pixel Axis 0 [x]:World 0'

    def test_nan_component(self):
        # regression test for case when all values are NaN in a component
        data = core.Data()
        data.add_component([np.nan, np.nan, np.nan], label='c1')
        self.data_collection.append(data)
        self.viewer.add_data(data)

    def test_density_map(self):

        self.viewer.add_data(self.data)

        density_artist = self.viewer.layers[0].density_artist
        if hasattr(density_artist, 'histogram2d_helper'):
            density_artist = density_artist.histogram2d_helper

        self.viewer.state.layers[0].points_mode = 'auto'
        assert len(density_artist._x) == 0
        self.viewer.state.layers[0].points_mode = 'density'
        assert len(density_artist._x) == 4
        self.viewer.state.layers[0].points_mode = 'markers'
        assert len(density_artist._x) == 0

    def test_density_map_color(self):

        # Regression test to make sure things don't crash when changing
        # back to markers if the color mode is cmap

        self.viewer.add_data(self.data)
        self.viewer.state.layers[0].points_mode = 'density'
        self.viewer.state.layers[0].cmap_mode = 'Linear'
        self.viewer.state.layers[0].size_mode = 'Linear'
        self.viewer.state.layers[0].points_mode = 'markers'
        self.viewer.state.layers[0].points_mode = 'density'

    @pytest.mark.skipif('PYSIDE2_INSTALLED')
    @pytest.mark.parametrize('protocol', [0, 1])
    def test_session_back_compat(self, protocol):

        filename = os.path.join(DATA, 'scatter_v{0}.glu'.format(protocol))

        with open(filename, 'r') as f:
            session = f.read()

        state = GlueUnSerializer.loads(session)

        ga = state.object('__main__')

        dc = ga.session.data_collection

        assert len(dc) == 1

        assert dc[0].label == 'basic'

        viewer1 = ga.viewers[0][0]
        assert len(viewer1.state.layers) == 3
        assert viewer1.state.x_att is dc[0].id['a']
        assert viewer1.state.y_att is dc[0].id['b']
        assert_allclose(viewer1.state.x_min, -1.04)
        assert_allclose(viewer1.state.x_max, 1.04)
        assert_allclose(viewer1.state.y_min, 1.98)
        assert_allclose(viewer1.state.y_max, 3.02)
        assert not viewer1.state.x_log
        assert not viewer1.state.y_log
        assert viewer1.state.layers[0].visible
        assert viewer1.state.layers[1].visible
        assert viewer1.state.layers[2].visible

        viewer2 = ga.viewers[0][1]
        assert len(viewer2.state.layers) == 3
        assert viewer2.state.x_att is dc[0].id['a']
        assert viewer2.state.y_att is dc[0].id['c']
        assert_allclose(viewer2.state.x_min, 9.5e-6)
        assert_allclose(viewer2.state.x_max, 1.05)
        assert_allclose(viewer2.state.y_min, 0.38)
        assert_allclose(viewer2.state.y_max, 5.25)
        assert viewer2.state.x_log
        assert viewer2.state.y_log
        assert viewer2.state.layers[0].visible
        assert not viewer2.state.layers[1].visible
        assert viewer2.state.layers[2].visible

        viewer3 = ga.viewers[0][2]
        assert len(viewer3.state.layers) == 3
        assert viewer3.state.x_att is dc[0].id['b']
        assert viewer3.state.y_att is dc[0].id['a']
        assert_allclose(viewer3.state.x_min, 0)
        assert_allclose(viewer3.state.x_max, 5)
        assert_allclose(viewer3.state.y_min, -5)
        assert_allclose(viewer3.state.y_max, 5)
        assert not viewer3.state.x_log
        assert not viewer3.state.y_log
        assert viewer3.state.layers[0].visible
        assert viewer3.state.layers[1].visible
        assert not viewer3.state.layers[2].visible

    def test_session_line_back_compat(self):

        # Backward-compatibility for v0.11 files in which the line and scatter
        # plots were defined as separate styles.

        filename = os.path.join(DATA, 'scatter_and_line_v1.glu')

        with open(filename, 'r') as f:
            session = f.read()

        state = GlueUnSerializer.loads(session)

        ga = state.object('__main__')

        dc = ga.session.data_collection

        assert len(dc) == 1

        assert dc[0].label == 'table'

        viewer1 = ga.viewers[0][0]
        assert len(viewer1.state.layers) == 1
        assert viewer1.state.x_att is dc[0].id['a']
        assert viewer1.state.y_att is dc[0].id['b']
        assert viewer1.state.layers[0].markers_visible
        assert not viewer1.state.layers[0].line_visible

        viewer1 = ga.viewers[0][1]
        assert len(viewer1.state.layers) == 1
        assert viewer1.state.x_att is dc[0].id['a']
        assert viewer1.state.y_att is dc[0].id['b']
        assert not viewer1.state.layers[0].markers_visible
        assert viewer1.state.layers[0].line_visible

    def test_save_svg(self, tmpdir):
        # Regression test for a bug in AxesCache that caused SVG saving to
        # fail (because renderer.buffer_rgba did not exist)
        self.viewer.add_data(self.data)
        filename = tmpdir.join('test.svg').strpath
        self.viewer.axes.figure.savefig(filename)

    def test_2d(self):

        viewer_state = self.viewer.state

        self.viewer.add_data(self.data_2d)

        assert viewer_state.x_att is self.data_2d.id['a']
        assert_allclose(viewer_state.x_min, 1 - 0.15)
        assert_allclose(viewer_state.x_max, 4 + 0.15)

        assert viewer_state.y_att is self.data_2d.id['b']
        assert_allclose(viewer_state.y_min, 5 - 0.15)
        assert_allclose(viewer_state.y_max, 8 + 0.15)

        assert self.viewer.layers[0].plot_artist.get_xdata().shape == (4,)

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

        d1 = Data(a=[1, 2, 3], label='d3')
        d2 = Data(b=[1, 2, 3], label='d4')
        d3 = Data(c=[1, 2, 3], label='d5')
        d4 = Data(d=[1, 2, 3], label='d6')

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

    @pytest.mark.parametrize('ndim', [1, 2])
    def test_all_options(self, ndim):

        # This test makes sure that all the code for the different scatter modes
        # gets run, though does not check the result.

        viewer_state = self.viewer.state

        if ndim == 1:
            data = self.data
        elif ndim == 2:
            data = self.data_2d

        self.viewer.add_data(data)

        layer_state = viewer_state.layers[0]

        layer_state.style = 'Scatter'

        layer_state.size_mode = 'Linear'
        layer_state.size_att = data.id['y']
        layer_state.size_vmin = 1.2
        layer_state.size_vmax = 4.
        layer_state.size_scaling = 2

        layer_state.cmap_mode = 'Linear'
        layer_state.cmap_att = data.id['x']
        layer_state.cmap_vmin = -1
        layer_state.cmap_vmax = 2.
        layer_state.cmap = colormaps.members[3][1]

        # Check inverting works
        layer_state.cmap_vmin = 3.

        layer_state.size_mode = 'Fixed'

        layer_state.xerr_visible = True
        layer_state.xerr_att = data.id['x']
        layer_state.yerr_visible = True
        layer_state.yerr_att = data.id['y']

        layer_state.style = 'Line'
        layer_state.linewidth = 3
        layer_state.linestyle = 'dashed'

    def test_session_categorical(self, tmpdir):

        def visible_xaxis_labels(ax):
            # Due to a bug in Matplotlib the labels returned outside the field
            # of view may be incorrect: https://github.com/matplotlib/matplotlib/issues/9397
            pos = ax.xaxis.get_ticklocs()
            labels = [tick.get_text() for tick in ax.xaxis.get_ticklabels()]
            xmin, xmax = ax.get_xlim()
            return [labels[i] for i in range(len(pos)) if pos[i] >= xmin and pos[i] <= xmax]

        # Regression test for a bug that caused a restored scatter viewer
        # with a categorical component to not show the categorical labels
        # as tick labels.

        filename = tmpdir.join('test_session_categorical.glu').strpath

        self.viewer.add_data(self.data)
        self.viewer.state.x_att = self.data.id['z']

        assert visible_xaxis_labels(self.viewer.axes) == ['a', 'b', 'c']

        self.session.application.save_session(filename)

        with open(filename, 'r') as f:
            session = f.read()

        state = GlueUnSerializer.loads(session)

        ga = state.object('__main__')

        dc = ga.session.data_collection

        viewer = ga.viewers[0][0]
        assert viewer.state.x_att is dc[0].id['z']
        assert visible_xaxis_labels(self.viewer.axes) == ['a', 'b', 'c']

    def test_enable_disable_components_combo(self):

        # Regression test for a bug that caused an error when turning off pixel
        # components from combo boxes.

        self.viewer.add_data(self.data)

        self.data['a'] = self.data.id['x'] + 5

        self.viewer.state.x_att_helper.pixel_coord = True

        self.viewer.state.x_att = self.data.pixel_component_ids[0]

        self.viewer.state.x_att_helper.pixel_coord = False

    def test_datetime64_support(self, tmpdir):

        self.data.add_component(np.array([100, 200, 300, 400], dtype='M8[D]'), 't1')
        self.data.add_component(np.array([200, 300, 400, 500], dtype='M8[D]'), 't2')
        self.viewer.add_data(self.data)
        self.viewer.state.x_att = self.data.id['t1']
        self.viewer.state.y_att = self.data.id['y']

        # Matplotlib deals with dates by converting them to the number of days
        # since 01-01-0001, so we can check that the limits are correctly
        # converted (and not 100 to 400)
        assert self.viewer.axes.get_xlim() == (719248.0, 719578.0)
        assert self.viewer.axes.get_ylim() == (3.2 - 0.015, 3.5 + 0.015)

        # Apply an ROI selection in plotting coordinates
        roi = RectangularROI(xmin=719313, xmax=719513, ymin=3, ymax=4)
        self.viewer.apply_roi(roi)

        # Check that the two middle elements are selected
        assert_equal(self.data.subsets[0].to_mask(), [0, 1, 1, 0])

        # Now do the same with the y axis
        self.viewer.state.y_att = self.data.id['t2']

        assert self.viewer.axes.get_xlim() == (719248.0, 719578.0)
        assert self.viewer.axes.get_ylim() == (719348.0, 719678.0)

        # Apply an ROI selection in plotting coordinates
        edit = self.session.edit_subset_mode
        edit.edit_subset = []
        roi = CircularROI(xc=719463, yc=719563, radius=200)
        self.viewer.apply_roi(roi)
        assert_equal(self.data.subsets[1].to_mask(), [0, 1, 1, 1])

        # Make sure that the Qt labels look ok
        self.viewer.state.y_att = self.data.id['y']
        options = self.viewer.options_widget().ui
        assert options.valuetext_x_min.text() == '1970-03-27'
        assert options.valuetext_x_max.text() == '1971-02-20'
        assert options.valuetext_y_min.text() == '3.185'
        assert options.valuetext_y_max.text() == '3.515'

        # Make sure that we can set the xmin/xmax to a string date
        assert_equal(self.viewer.state.x_min, np.datetime64('1970-03-27', 'D'))
        options.valuetext_x_min.setText('1970-04-14')
        options.valuetext_x_min.editingFinished.emit()
        assert self.viewer.axes.get_xlim() == (719266.0, 719578.0)
        assert_equal(self.viewer.state.x_min, np.datetime64('1970-04-14', 'D'))

        # Make sure that everything works fine after saving/reloading
        filename = tmpdir.join('test_datetime64.glu').strpath
        self.session.application.save_session(filename)
        with open(filename, 'r') as f:
            session = f.read()
        state = GlueUnSerializer.loads(session)
        ga = state.object('__main__')
        viewer = ga.viewers[0][0]
        options = viewer.options_widget().ui

        assert_equal(self.viewer.state.x_min, np.datetime64('1970-04-14', 'D'))

        assert options.valuetext_x_min.text() == '1970-04-14'
        assert options.valuetext_x_max.text() == '1971-02-20'
        assert options.valuetext_y_min.text() == '3.185'
        assert options.valuetext_y_max.text() == '3.515'
