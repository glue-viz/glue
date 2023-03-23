# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

import os
from collections import Counter

import pytest
import numpy as np

from numpy.testing import assert_allclose, assert_equal

from glue.config import colormaps
from glue.core.message import SubsetUpdateMessage
from glue.core import HubListener, Data
from glue.core.roi import XRangeROI, RectangularROI, CircularROI
from glue.core.roi_pretransforms import FullSphereLongitudeTransform, ProjectionMplTransform, RadianTransform
from glue.core.subset import RoiSubsetState, AndState
from glue import core
from glue.core.component_id import ComponentID
from glue.utils.qt import combo_as_string, process_events
from glue.viewers.matplotlib.qt.tests.test_data_viewer import BaseTestMatplotlibDataViewer
from glue.core.state import GlueUnSerializer
from glue.app.qt.layer_tree_widget import LayerTreeWidget
from glue.app.qt import GlueApplication

from ..data_viewer import ScatterViewer

DATA = os.path.join(os.path.dirname(__file__), 'data')

fullsphere_projections = ['aitoff', 'hammer', 'lambert', 'mollweide']


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
        self.data_fullsphere = Data(label='d3', x=[6.9, -1.1, 1.2, -3.7],
                                    y=[-0.2, 1.0, 0.5, -1.1])

        self.app = GlueApplication()
        self.session = self.app.session
        self.hub = self.session.hub

        self.data_collection = self.session.data_collection
        self.data_collection.append(self.data)
        self.data_collection.append(self.data_2d)
        self.data_collection.append(self.data_fullsphere)

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

        assert combo_as_string(self.viewer.options_widget().ui.combosel_x_att) == 'Main components:x:y:z:Coordinate components:Pixel Axis 0 [x]'
        assert combo_as_string(self.viewer.options_widget().ui.combosel_y_att) == 'Main components:x:y:z:Coordinate components:Pixel Axis 0 [x]'

        assert viewer_state.x_att is self.data.id['x']
        assert_allclose(viewer_state.x_min, -1.1 - 0.18)
        assert_allclose(viewer_state.x_max, 3.4 + 0.18)

        assert viewer_state.y_att is self.data.id['y']
        assert_allclose(viewer_state.y_min, 3.2 - 0.012)
        assert_allclose(viewer_state.y_max, 3.5 + 0.012)

        assert not viewer_state.x_log
        assert not viewer_state.y_log

        assert len(viewer_state.layers) == 1

        # Change to categorical component and check new values

        viewer_state.y_att = self.data.id['z']

        assert viewer_state.x_att is self.data.id['x']
        assert_allclose(viewer_state.x_min, -1.1 - 0.18)
        assert_allclose(viewer_state.x_max, 3.4 + 0.18)

        assert viewer_state.y_att is self.data.id['z']
        assert_allclose(viewer_state.y_min, -0.5 - 0.12)
        assert_allclose(viewer_state.y_max, 2.5 + 0.12)

        assert not viewer_state.x_log
        assert not viewer_state.y_log

    def test_flip(self):

        viewer_state = self.viewer.state

        self.viewer.add_data(self.data)

        assert_allclose(viewer_state.x_min, -1.1 - 0.18)
        assert_allclose(viewer_state.x_max, 3.4 + 0.18)

        self.viewer.options_widget().button_flip_x.click()

        assert_allclose(viewer_state.x_max, -1.1 - 0.18)
        assert_allclose(viewer_state.x_min, 3.4 + 0.18)

        assert_allclose(viewer_state.y_min, 3.2 - 0.012)
        assert_allclose(viewer_state.y_max, 3.5 + 0.012)

        self.viewer.options_widget().button_flip_y.click()

        assert_allclose(viewer_state.y_max, 3.2 - 0.012)
        assert_allclose(viewer_state.y_min, 3.5 + 0.012)

    def test_remove_data(self):
        self.viewer.add_data(self.data)
        assert combo_as_string(self.viewer.options_widget().ui.combosel_x_att) == 'Main components:x:y:z:Coordinate components:Pixel Axis 0 [x]'
        assert combo_as_string(self.viewer.options_widget().ui.combosel_y_att) == 'Main components:x:y:z:Coordinate components:Pixel Axis 0 [x]'
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
        assert combo_as_string(self.viewer.options_widget().ui.combosel_x_att) == 'Main components:x:y:z:a:Coordinate components:Pixel Axis 0 [x]'
        assert combo_as_string(self.viewer.options_widget().ui.combosel_y_att) == 'Main components:x:y:z:a:Coordinate components:Pixel Axis 0 [x]'

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

        assert self.viewer.axes.get_xlabel() == 'x'
        assert self.viewer.axes.get_ylabel() == 'y'

        viewer_state.x_att = self.data.id['y']

        assert self.viewer.axes.get_xlabel() == 'y'
        assert self.viewer.axes.get_ylabel() == 'y'

        viewer_state.y_log = True

        assert self.viewer.axes.get_xlabel() == 'y'
        assert self.viewer.axes.get_ylabel() == 'y'

    def test_component_replaced(self):

        # regression test for 508 - if a component ID is replaced, we should
        # make sure that the component ID is selected if the old component ID
        # was selected

        self.viewer.add_data(self.data)
        self.viewer.state.x_att = self.data.id['x']
        test = ComponentID('test')
        self.data.update_id(self.viewer.state.x_att, test)
        assert self.viewer.state.x_att is test
        assert combo_as_string(self.viewer.options_widget().ui.combosel_x_att) == 'Main components:test:y:z:Coordinate components:Pixel Axis 0 [x]'

    def test_nan_component(self):
        # regression test for case when all values are NaN in a component
        data = core.Data()
        data.add_component([np.nan, np.nan, np.nan], label='c1')
        self.data_collection.append(data)
        self.viewer.add_data(data)

    def test_density_map(self):

        kwargs = dict(range=[(-5, 5), (-5, 5)], bins=(2, 2))

        self.viewer.add_data(self.data)
        self.viewer.state.layers[0].points_mode = 'auto'
        assert self.viewer.layers[0].state.compute_density_map(**kwargs).sum() == 0
        self.viewer.state.layers[0].points_mode = 'density'
        assert self.viewer.layers[0].state.compute_density_map(**kwargs).sum() == 4
        self.viewer.state.layers[0].points_mode = 'markers'
        assert self.viewer.layers[0].state.compute_density_map(**kwargs).sum() == 0

    def test_density_map_color(self):

        # Regression test to make sure things don't crash when changing
        # back to markers if the color mode is cmap

        self.viewer.add_data(self.data)
        self.viewer.state.layers[0].points_mode = 'density'
        self.viewer.state.layers[0].cmap_mode = 'Linear'
        self.viewer.state.layers[0].size_mode = 'Linear'
        self.viewer.state.layers[0].points_mode = 'markers'
        self.viewer.state.layers[0].points_mode = 'density'

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

        ga.close()

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

        ga.close()

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
        assert_allclose(viewer_state.x_min, 1 - 0.12)
        assert_allclose(viewer_state.x_max, 4 + 0.12)

        assert viewer_state.y_att is self.data_2d.id['b']
        assert_allclose(viewer_state.y_min, 5 - 0.12)
        assert_allclose(viewer_state.y_max, 8 + 0.12)

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

        process_events()

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

        ga.close()

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
        assert self.viewer.axes.get_xlim() == (719251.0, 719575.0)
        assert self.viewer.axes.get_ylim() == (3.2 - 0.012, 3.5 + 0.012)

        # Apply an ROI selection in plotting coordinates
        roi = RectangularROI(xmin=719313, xmax=719513, ymin=3, ymax=4)
        self.viewer.apply_roi(roi)

        # Check that the two middle elements are selected
        assert_equal(self.data.subsets[0].to_mask(), [0, 1, 1, 0])

        # Now do the same with the y axis
        self.viewer.state.y_att = self.data.id['t2']

        assert self.viewer.axes.get_xlim() == (719251.0, 719575.0)
        assert self.viewer.axes.get_ylim() == (719351.0, 719675.0)

        # Apply an ROI selection in plotting coordinates
        edit = self.session.edit_subset_mode
        edit.edit_subset = []
        roi = CircularROI(xc=719463, yc=719563, radius=200)
        self.viewer.apply_roi(roi)
        assert_equal(self.data.subsets[1].to_mask(), [0, 1, 1, 1])

        # Make sure that the Qt labels look ok
        self.viewer.state.y_att = self.data.id['y']
        options = self.viewer.options_widget().ui
        assert options.valuetext_x_min.text() == '1970-03-30'
        assert options.valuetext_x_max.text() == '1971-02-17'
        assert options.valuetext_y_min.text() == '3.188'
        assert options.valuetext_y_max.text() == '3.512'

        # Make sure that we can set the xmin/xmax to a string date
        assert_equal(self.viewer.state.x_min, np.datetime64('1970-03-30', 'D'))
        options.valuetext_x_min.setText('1970-04-14')
        options.valuetext_x_min.editingFinished.emit()
        assert self.viewer.axes.get_xlim() == (719266.0, 719575.0)
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
        assert options.valuetext_x_max.text() == '1971-02-17'
        assert options.valuetext_y_min.text() == '3.188'
        assert options.valuetext_y_max.text() == '3.512'

        ga.close()

    def test_datetime64_disabled(self, capsys):

        # Make sure that datetime components aren't options for the vector and
        # error markers.

        data = Data(label='test')
        data.add_component(np.array([100, 200, 300, 400], dtype='M8[D]'), 't1')
        data.add_component(np.array([200, 300, 400, 500], dtype='M8[D]'), 't2')
        data.add_component(np.array([200., 300., 400., 500.]), 'x')
        data.add_component(np.array([200., 300., 400., 500.]), 'y')
        self.data_collection.append(data)

        self.viewer.add_data(data)
        self.viewer.state.x_att = data.id['x']
        self.viewer.state.y_att = data.id['y']
        self.viewer.state.layers[0].cmap_mode = 'Linear'
        self.viewer.state.layers[0].cmap_att = data.id['x']
        self.viewer.state.layers[0].size_mode = 'Linear'
        self.viewer.state.layers[0].size_att = data.id['y']
        self.viewer.state.layers[0].vector_visible = True
        self.viewer.state.layers[0].xerr_visible = True
        self.viewer.state.layers[0].yerr_visible = True

        process_events()

        self.viewer.state.x_att = data.id['t1']
        self.viewer.state.y_att = data.id['t2']

        process_events()

        #  We use capsys here because the # error is otherwise only apparent in stderr.
        out, err = capsys.readouterr()
        assert out.strip() == ""
        assert err.strip() == ""

    def test_density_map_incompatible_subset(self, capsys):

        # Regression test for a bug that caused the scatter viewer to crash
        # if subset for density map was incompatible.

        data2 = Data(label='d1', x=[3.4, 2.3, -1.1, 0.3], y=[3.2, 3.3, 3.4, 3.5], z=['a', 'b', 'c', 'a'])

        self.data_collection.append(data2)

        self.viewer.add_data(self.data)
        self.viewer.add_data(data2)

        self.data_collection.new_subset_group('test', self.data.id['x'] > 1)

        for layer in self.viewer.state.layers:
            layer.density_map = True

        self.viewer.figure.canvas.draw()
        process_events()

        assert self.viewer.layers[0].enabled
        assert not self.viewer.layers[1].enabled
        assert self.viewer.layers[2].enabled
        assert not self.viewer.layers[3].enabled

    def test_density_map_line_error_vector(self, capsys):

        # Make sure that we don't allow/show lines/errors/vectors
        # if in density map mode.

        self.viewer.add_data(self.data)

        self.viewer.state.layers[0].line_visible = True
        self.viewer.state.layers[0].xerr_visible = True
        self.viewer.state.layers[0].yerr_visible = True
        self.viewer.state.layers[0].vector_visible = True

        # Setting density_map to True resets the visibility of
        # lines/errors/vectors.
        self.viewer.state.layers[0].density_map = True
        assert not self.viewer.state.layers[0].line_visible
        assert not self.viewer.state.layers[0].xerr_visible
        assert not self.viewer.state.layers[0].yerr_visible
        assert not self.viewer.state.layers[0].vector_visible

    def test_legend(self):
        viewer_state = self.viewer.state

        self.viewer.add_data(self.data)
        viewer_state.legend.visible = True

        handles, labels, handler_dict = self.viewer.get_handles_legend()
        assert len(handles) == 1
        assert labels[0] == 'd1'

        self.data_collection.new_subset_group('test', self.data.id['x'] > 1)
        assert len(viewer_state.layers) == 2
        handles, labels, handler_dict = self.viewer.get_handles_legend()

        assert len(handles) == 2
        assert labels[1] == 'test'
        print(handles[1][0])
        # assert handles[1][0].get_color() == viewer_state.layers[1].state.color

        # Add a non visible layer
        data2 = Data(label='d2', x=[3.4, 2.3, -1.1, 0.3], y=[3.2, 3.3, 3.4, 3.5])
        self.data_collection.append(data2)

        self.viewer.add_data(data2)
        assert len(viewer_state.layers) == 4

        # 'd2' is not enabled (no linked component)
        handles, labels, handler_dict = self.viewer.get_handles_legend()
        assert len(handles) == 2

    def test_changing_plot_modes(self):
        viewer_state = self.viewer.state
        viewer_state.plot_mode = 'polar'
        assert 'polar' in str(type(self.viewer.axes)).lower()
        viewer_state.plot_mode = 'aitoff'
        assert 'aitoff' in str(type(self.viewer.axes)).lower()
        viewer_state.plot_mode = 'hammer'
        assert 'hammer' in str(type(self.viewer.axes)).lower()
        viewer_state.plot_mode = 'lambert'
        assert 'lambert' in str(type(self.viewer.axes)).lower()
        viewer_state.plot_mode = 'mollweide'
        assert 'mollweide' in str(type(self.viewer.axes)).lower()

    # For the time being, the polar plots don't support log scaling
    # def test_limit_log_set_polar(self):
    #     self.viewer.add_data(self.data)
    #     viewer_state = self.viewer.state
    #     viewer_state.plot_mode = "polar"
    #     axes = self.viewer.axes
    #
    #     viewer_state.x_min = 0.5
    #     viewer_state.x_max = 1.5
    #     assert_allclose(axes.get_xlim(), [0.5, 1.5])
    #
    #     viewer_state.y_min = -2.5
    #     viewer_state.y_max = 2.5
    #     assert_allclose(axes.get_ylim(), [-2.5, 2.5])
    #
    #     viewer_state.y_log = True
    #     assert axes.get_yscale() == 'log'

    def test_limit_set_fullsphere(self):
        # Make sure that the full-sphere projections ignore instead of throwing exceptions
        self.viewer.add_data(self.data)
        viewer_state = self.viewer.state

        for proj in fullsphere_projections:
            viewer_state.plot_mode = proj
            error_msg = 'Issue with {} projection'.format(proj)
            axes = self.viewer.axes
            viewer_state.x_min = 0.5
            viewer_state.x_max = 1.5
            viewer_state.y_min = -2.5
            viewer_state.y_max = 2.5
            assert_allclose(axes.get_xlim(), [-np.pi, np.pi], err_msg=error_msg)
            assert_allclose(axes.get_ylim(), [-np.pi / 2, np.pi / 2], err_msg=error_msg)

    def test_changing_mode_limits(self):
        self.viewer.add_data(self.data)
        viewer_state = self.viewer.state
        old_xmin = viewer_state.x_min
        old_xmax = viewer_state.x_max
        old_ymin = viewer_state.y_min
        old_ymax = viewer_state.y_max
        # Make sure limits are reset first
        viewer_state.x_max += 3

        # Currently, when we change to polar mode, the x-limits are changed to 0, 2pi
        viewer_state.plot_mode = 'polar'
        assert_allclose(viewer_state.x_min, 0)
        assert_allclose(viewer_state.x_max, 2 * np.pi)
        assert_allclose(self.viewer.axes.get_xlim(), [0, 2 * np.pi])
        assert_allclose(viewer_state.y_min, old_ymin)
        assert_allclose(viewer_state.y_max, old_ymax)
        assert_allclose(self.viewer.axes.get_ylim(), [old_ymin, old_ymax])

        viewer_state.plot_mode = 'rectilinear'
        assert_allclose(viewer_state.x_min, old_xmin)
        assert_allclose(viewer_state.x_max, old_xmax)
        assert_allclose(self.viewer.axes.get_xlim(), [old_xmin, old_xmax])
        assert_allclose(viewer_state.y_min, old_ymin)
        assert_allclose(viewer_state.y_max, old_ymax)
        assert_allclose(self.viewer.axes.get_ylim(), [old_ymin, old_ymax])

        for proj in fullsphere_projections:
            viewer_state.plot_mode = 'rectilinear'
            viewer_state.plot_mode = proj
            error_msg = 'Issue with {} projection'.format(proj)
            assert_allclose(viewer_state.x_min, -np.pi)
            assert_allclose(viewer_state.x_max, np.pi)
            assert_allclose(self.viewer.axes.get_xlim(), [-np.pi, np.pi], err_msg=error_msg)
            assert_allclose(viewer_state.y_min, -np.pi / 2)
            assert_allclose(viewer_state.y_max, np.pi / 2)
            assert_allclose(self.viewer.axes.get_ylim(), [-np.pi / 2, np.pi / 2], err_msg=error_msg)

    def test_changing_mode_log(self):
        # Test to make sure we reset the log axes to false when changing modes to prevent problems
        self.viewer.add_data(self.data)
        viewer_state = self.viewer.state
        viewer_state.x_log = True
        viewer_state.y_log = True

        viewer_state.plot_mode = 'polar'
        assert not viewer_state.x_log
        assert not viewer_state.y_log
        assert self.viewer.axes.get_xscale() == 'linear'
        assert self.viewer.axes.get_yscale() == 'linear'
        viewer_state.y_log = True

        viewer_state.plot_mode = 'rectilinear'
        assert not viewer_state.x_log
        assert not viewer_state.y_log
        assert self.viewer.axes.get_xscale() == 'linear'
        assert self.viewer.axes.get_yscale() == 'linear'

        for proj in fullsphere_projections:
            viewer_state.plot_mode = 'rectilinear'
            viewer_state.x_log = True
            viewer_state.y_log = True
            viewer_state.plot_mode = proj
            error_msg = 'Issue with {} projection'.format(proj)
            assert not viewer_state.x_log, error_msg
            assert not viewer_state.y_log, error_msg
            assert self.viewer.axes.get_xscale() == 'linear', error_msg
            assert self.viewer.axes.get_yscale() == 'linear', error_msg

    def test_full_circle_utility(self):
        # Make sure that the full circle function behaves well
        self.viewer.add_data(self.data)
        viewer_state = self.viewer.state
        old_xmin = viewer_state.x_min
        old_xmax = viewer_state.x_max
        old_ymin = viewer_state.y_min
        old_ymax = viewer_state.y_max
        viewer_state.full_circle()
        assert_allclose([viewer_state.x_min, viewer_state.x_max], [old_xmin, old_xmax])
        assert_allclose([viewer_state.y_min, viewer_state.y_max], [old_ymin, old_ymax])

        viewer_state.plot_mode = 'polar'
        viewer_state.full_circle()
        assert_allclose([viewer_state.x_min, viewer_state.x_max], [0, 2 * np.pi])
        assert_allclose([viewer_state.y_min, viewer_state.y_max], [old_ymin, old_ymax])

        for proj in fullsphere_projections:
            error_msg = 'Issue with {} projection'.format(proj)
            viewer_state.plot_mode = proj
            viewer_state.full_circle()
            assert_allclose([viewer_state.x_min, viewer_state.x_max], [-np.pi, np.pi], err_msg=error_msg)
            assert_allclose([viewer_state.y_min, viewer_state.y_max], [-np.pi / 2, np.pi / 2], err_msg=error_msg)

    def test_limits_log_widget_polar_cartesian(self):
        ui = self.viewer.options_widget().ui
        viewer_state = self.viewer.state
        viewer_state.plot_mode = 'polar'
        assert not ui.bool_x_log.isEnabled()
        assert not ui.bool_x_log_.isEnabled()
        assert not ui.bool_y_log.isEnabled()
        assert not ui.bool_y_log_.isEnabled()
        assert ui.valuetext_x_min.isEnabled()
        assert ui.button_flip_x.isEnabled()
        assert ui.valuetext_x_max.isEnabled()
        assert ui.valuetext_y_min.isEnabled()
        assert ui.button_flip_y.isEnabled()
        assert ui.valuetext_y_max.isEnabled()
        assert ui.button_full_circle.isHidden()

        viewer_state.plot_mode = 'rectilinear'
        assert ui.bool_x_log.isEnabled()
        assert ui.bool_x_log_.isEnabled()
        assert ui.bool_y_log.isEnabled()
        assert ui.bool_y_log_.isEnabled()
        assert ui.valuetext_x_min.isEnabled()
        assert ui.button_flip_x.isEnabled()
        assert ui.valuetext_x_max.isEnabled()
        assert ui.valuetext_y_min.isEnabled()
        assert ui.button_flip_y.isEnabled()
        assert ui.valuetext_y_max.isEnabled()
        assert ui.button_full_circle.isHidden()
        assert ui.button_full_circle.isHidden()

    def test_limits_log_widget_fullsphere(self):
        ui = self.viewer.options_widget().ui
        viewer_state = self.viewer.state
        for proj in fullsphere_projections:
            error_msg = 'Issue with {} projection'.format(proj)
            viewer_state.plot_mode = proj
            not ui.bool_x_log.isEnabled()
            assert not ui.bool_x_log_.isEnabled(), error_msg
            assert not ui.bool_y_log.isEnabled(), error_msg
            assert not ui.bool_y_log_.isEnabled(), error_msg
            assert not ui.valuetext_x_min.isEnabled(), error_msg
            assert ui.button_flip_x.isEnabled(), error_msg
            assert not ui.valuetext_x_max.isEnabled(), error_msg
            assert not ui.valuetext_y_min.isEnabled(), error_msg
            assert ui.button_flip_y.isEnabled(), error_msg
            assert not ui.valuetext_y_max.isEnabled(), error_msg
            assert ui.button_full_circle.isHidden(), error_msg

            viewer_state.plot_mode = 'rectilinear'
            assert ui.bool_x_log.isEnabled()
            assert ui.bool_x_log_.isEnabled()
            assert ui.bool_y_log.isEnabled()
            assert ui.bool_y_log_.isEnabled()
            assert ui.valuetext_x_min.isEnabled()
            assert ui.button_flip_x.isEnabled()
            assert ui.valuetext_x_max.isEnabled()
            assert ui.valuetext_y_min.isEnabled()
            assert ui.button_flip_y.isEnabled()
            assert ui.valuetext_y_max.isEnabled()
            assert ui.button_full_circle.isHidden()

    @pytest.mark.parametrize('angle_unit,expected_mask', [('radians', [0, 0, 0, 1]), ('degrees', [1, 1, 0, 1])])
    def test_apply_roi_polar(self, angle_unit, expected_mask):
        self.viewer.add_data(self.data)
        viewer_state = self.viewer.state
        roi = RectangularROI(0.5, 1, 0.5, 1)
        viewer_state.plot_mode = 'polar'
        viewer_state.full_circle()
        assert len(self.viewer.layers) == 1

        viewer_state.angle_unit = angle_unit

        self.viewer.apply_roi(roi)

        assert len(self.viewer.layers) == 2
        assert len(self.data.subsets) == 1

        assert_allclose(self.data.subsets[0].to_mask(), expected_mask)

        state = self.data.subsets[0].subset_state
        assert isinstance(state, RoiSubsetState)
        assert state.pretransform
        pretrans = state.pretransform
        if angle_unit == 'radians':
            assert isinstance(pretrans, ProjectionMplTransform)
            projtrans = pretrans
        elif angle_unit == 'degrees':
            assert isinstance(pretrans, RadianTransform)
            projtrans = pretrans._next_transform
            assert isinstance(projtrans, ProjectionMplTransform)
        assert projtrans._state['projection'] == 'polar'
        assert_allclose(projtrans._state['x_lim'], [viewer_state.x_min, viewer_state.x_max])
        assert_allclose(projtrans._state['y_lim'], [viewer_state.y_min, viewer_state.y_max])
        assert projtrans._state['x_scale'] == 'linear'
        assert projtrans._state['y_scale'] == 'linear'
        self.data.subsets[0].delete()

        viewer_state.y_log = True
        self.viewer.apply_roi(roi)
        state = self.data.subsets[0].subset_state
        assert state.pretransform
        pretrans = state.pretransform
        if angle_unit == 'radians':
            assert isinstance(pretrans, ProjectionMplTransform)
            projtrans = pretrans
        elif angle_unit == 'degrees':
            assert isinstance(pretrans, RadianTransform)
            projtrans = pretrans._next_transform
            assert isinstance(projtrans, ProjectionMplTransform)
        assert projtrans._state['y_scale'] == 'log'
        viewer_state.y_log = False

    @pytest.mark.parametrize('angle_unit,expected_mask', [('radians', [1, 0, 0, 1]), ('degrees', [1, 0, 0, 0])])
    def test_apply_roi_fullsphere(self, angle_unit, expected_mask):
        self.viewer.add_data(self.data_fullsphere)
        viewer_state = self.viewer.state
        roi = RectangularROI(0.5, 1, 0, 0.5)

        viewer_state.angle_unit = angle_unit
        for proj in fullsphere_projections:
            viewer_state.plot_mode = proj
            assert len(self.viewer.layers) == 1

            self.viewer.apply_roi(roi)

            assert len(self.viewer.layers) == 2
            assert len(self.data_fullsphere.subsets) == 1

            subset = self.data_fullsphere.subsets[0]
            state = subset.subset_state
            assert isinstance(state, RoiSubsetState)

            assert state.pretransform
            pretrans = state.pretransform
            if angle_unit == 'degrees':
                assert isinstance(pretrans, RadianTransform)
                pretrans = pretrans._next_transform
            assert isinstance(pretrans, FullSphereLongitudeTransform)
            projtrans = pretrans._next_transform
            assert isinstance(projtrans, ProjectionMplTransform)

            assert_allclose(subset.to_mask(), expected_mask)

            assert projtrans._state['projection'] == proj
            assert_allclose(projtrans._state['x_lim'], [viewer_state.x_min, viewer_state.x_max])
            assert_allclose(projtrans._state['y_lim'], [viewer_state.y_min, viewer_state.y_max])
            assert projtrans._state['x_scale'] == 'linear'
            assert projtrans._state['y_scale'] == 'linear'
            subset.delete()
