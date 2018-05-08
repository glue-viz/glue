# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import os
from collections import Counter

import pytest
import numpy as np

from numpy.testing import assert_equal, assert_allclose

from glue.core.message import SubsetUpdateMessage
from glue.core import HubListener, Data
from glue.core.roi import XRangeROI
from glue.core.subset import RangeSubsetState, CategoricalROISubsetState
from glue import core
from glue.app.qt import GlueApplication
from glue.core.component_id import ComponentID
from glue.utils.qt import combo_as_string
from glue.viewers.matplotlib.qt.tests.test_data_viewer import BaseTestMatplotlibDataViewer
from glue.core.state import GlueUnSerializer
from glue.app.qt.layer_tree_widget import LayerTreeWidget

from ..data_viewer import HistogramViewer

DATA = os.path.join(os.path.dirname(__file__), 'data')


class TestHistogramCommon(BaseTestMatplotlibDataViewer):
    def init_data(self):
        return Data(label='d1', x=[3.4, 2.3, -1.1, 0.3], y=['a', 'b', 'c', 'a'])
    viewer_cls = HistogramViewer


class TestHistogramViewer(object):

    def setup_method(self, method):

        self.data = Data(label='d1', x=[3.4, 2.3, -1.1, 0.3], y=['a', 'b', 'c', 'a'])

        self.app = GlueApplication()
        self.session = self.app.session
        self.hub = self.session.hub

        self.data_collection = self.session.data_collection
        self.data_collection.append(self.data)

        self.viewer = self.app.new_data_viewer(HistogramViewer)

    def teardown_method(self, method):
        self.viewer.close()
        self.viewer = None
        self.app.close()
        self.app = None

    def test_basic(self):

        viewer_state = self.viewer.state

        # Check defaults when we add data
        self.viewer.add_data(self.data)

        assert combo_as_string(self.viewer.options_widget().ui.combosel_x_att) == 'Main components:x:y:Coordinate components:Pixel Axis 0 [x]:World 0'

        assert viewer_state.x_att is self.data.id['x']
        assert viewer_state.x_min == -1.1
        assert viewer_state.x_max == 3.4
        assert viewer_state.y_min == 0.0
        assert viewer_state.y_max == 1.2

        assert viewer_state.hist_x_min == -1.1
        assert viewer_state.hist_x_max == 3.4
        assert viewer_state.hist_n_bin == 15

        assert not viewer_state.cumulative
        assert not viewer_state.normalize

        assert not viewer_state.x_log
        assert not viewer_state.y_log

        assert len(viewer_state.layers) == 1

        # Change to categorical component and check new values

        viewer_state.x_att = self.data.id['y']

        assert viewer_state.x_min == -0.5
        assert viewer_state.x_max == 2.5
        assert viewer_state.y_min == 0.0
        assert viewer_state.y_max == 2.4

        assert viewer_state.hist_x_min == -0.5
        assert viewer_state.hist_x_max == 2.5
        assert viewer_state.hist_n_bin == 3

        assert not viewer_state.cumulative
        assert not viewer_state.normalize

        assert not viewer_state.x_log
        assert not viewer_state.y_log

    def test_log_labels(self):

        # Regression test to make sure the labels are correctly changed to log
        # when the x-axis is in log space.

        viewer_state = self.viewer.state
        data = Data(x=np.logspace(-5, 5, 10000))
        self.data_collection.append(data)

        self.viewer.add_data(data)
        viewer_state.x_log = True

        labels = [x.get_text() for x in self.viewer.axes.xaxis.get_ticklabels()]

        # Different Matplotlib versions return slightly different
        # labels, but the ones below should be present regardless
        # of Matplotlib version.
        expected_present = ['$\\mathdefault{10^{-5}}$',
                            '$\\mathdefault{10^{-3}}$',
                            '$\\mathdefault{10^{-1}}$',
                            '$\\mathdefault{10^{1}}$',
                            '$\\mathdefault{10^{3}}$',
                            '$\\mathdefault{10^{5}}$']

        for label in expected_present:
            assert label in labels

    def test_flip(self):

        viewer_state = self.viewer.state

        self.viewer.add_data(self.data)

        assert viewer_state.x_min == -1.1
        assert viewer_state.x_max == 3.4

        self.viewer.options_widget().button_flip_x.click()

        assert viewer_state.x_min == 3.4
        assert viewer_state.x_max == -1.1

    def test_remove_data(self):
        self.viewer.add_data(self.data)
        assert combo_as_string(self.viewer.options_widget().ui.combosel_x_att) == 'Main components:x:y:Coordinate components:Pixel Axis 0 [x]:World 0'
        self.data_collection.remove(self.data)
        assert combo_as_string(self.viewer.options_widget().ui.combosel_x_att) == ''

    def test_update_component_updates_title(self):
        self.viewer.add_data(self.data)
        assert self.viewer.windowTitle() == '1D Histogram'
        self.viewer.state.x_att = self.data.id['y']
        assert self.viewer.windowTitle() == '1D Histogram'

    def test_combo_updates_with_component_add(self):
        self.viewer.add_data(self.data)
        self.data.add_component([3, 4, 1, 2], 'z')
        assert self.viewer.state.x_att is self.data.id['x']
        assert combo_as_string(self.viewer.options_widget().ui.combosel_x_att) == 'Main components:x:y:z:Coordinate components:Pixel Axis 0 [x]:World 0'

    def test_nonnumeric_first_component(self):
        # regression test for #208. Shouldn't complain if
        # first component is non-numerical
        data = core.Data()
        data.add_component(['a', 'b', 'c'], label='c1')
        data.add_component([1, 2, 3], label='c2')
        self.data_collection.append(data)
        self.viewer.add_data(data)

    def test_nan_component(self):
        # regression test for case when all values are NaN in a component
        data = core.Data()
        data.add_component([np.nan, np.nan, np.nan], label='c1')
        self.data_collection.append(data)
        self.viewer.add_data(data)

    def test_histogram_values(self):

        # Check the actual values of the histograms

        viewer_state = self.viewer.state

        self.viewer.add_data(self.data)

        # Numerical attribute

        viewer_state.hist_x_min = -5
        viewer_state.hist_x_max = 5
        viewer_state.hist_n_bin = 4

        assert_allclose(self.viewer.state.y_max, 2.4)

        assert_allclose(self.viewer.layers[0].mpl_hist, [0, 1, 2, 1])
        assert_allclose(self.viewer.layers[0].mpl_bins, [-5, -2.5, 0, 2.5, 5])

        cid = self.data.visible_components[0]
        self.data_collection.new_subset_group('subset 1', cid < 2)

        assert_allclose(self.viewer.layers[1].mpl_hist, [0, 1, 1, 0])
        assert_allclose(self.viewer.layers[1].mpl_bins, [-5, -2.5, 0, 2.5, 5])

        viewer_state.normalize = True

        assert_allclose(self.viewer.state.y_max, 0.24)
        assert_allclose(self.viewer.layers[0].mpl_hist, [0, 0.1, 0.2, 0.1])
        assert_allclose(self.viewer.layers[0].mpl_bins, [-5, -2.5, 0, 2.5, 5])
        assert_allclose(self.viewer.layers[1].mpl_hist, [0, 0.2, 0.2, 0])
        assert_allclose(self.viewer.layers[1].mpl_bins, [-5, -2.5, 0, 2.5, 5])

        viewer_state.cumulative = True

        assert_allclose(self.viewer.state.y_max, 1.2)
        assert_allclose(self.viewer.layers[0].mpl_hist, [0, 0.25, 0.75, 1.0])
        assert_allclose(self.viewer.layers[0].mpl_bins, [-5, -2.5, 0, 2.5, 5])
        assert_allclose(self.viewer.layers[1].mpl_hist, [0, 0.5, 1.0, 1.0])
        assert_allclose(self.viewer.layers[1].mpl_bins, [-5, -2.5, 0, 2.5, 5])

        viewer_state.normalize = False

        assert_allclose(self.viewer.state.y_max, 4.8)
        assert_allclose(self.viewer.layers[0].mpl_hist, [0, 1, 3, 4])
        assert_allclose(self.viewer.layers[0].mpl_bins, [-5, -2.5, 0, 2.5, 5])
        assert_allclose(self.viewer.layers[1].mpl_hist, [0, 1, 2, 2])
        assert_allclose(self.viewer.layers[1].mpl_bins, [-5, -2.5, 0, 2.5, 5])

        viewer_state.cumulative = False

        # Categorical attribute

        viewer_state.x_att = self.data.id['y']

        formatter = self.viewer.axes.xaxis.get_major_formatter()
        xlabels = [formatter.format_data(pos) for pos in range(3)]
        assert xlabels == ['a', 'b', 'c']

        assert_allclose(self.viewer.state.y_max, 2.4)
        assert_allclose(self.viewer.layers[0].mpl_hist, [2, 1, 1])
        assert_allclose(self.viewer.layers[0].mpl_bins, [-0.5, 0.5, 1.5, 2.5])
        assert_allclose(self.viewer.layers[1].mpl_hist, [1, 0, 1])
        assert_allclose(self.viewer.layers[1].mpl_bins, [-0.5, 0.5, 1.5, 2.5])

        viewer_state.normalize = True

        assert_allclose(self.viewer.state.y_max, 0.6)
        assert_allclose(self.viewer.layers[0].mpl_hist, [0.5, 0.25, 0.25])
        assert_allclose(self.viewer.layers[0].mpl_bins, [-0.5, 0.5, 1.5, 2.5])
        assert_allclose(self.viewer.layers[1].mpl_hist, [0.5, 0, 0.5])
        assert_allclose(self.viewer.layers[1].mpl_bins, [-0.5, 0.5, 1.5, 2.5])

        viewer_state.cumulative = True

        assert_allclose(self.viewer.state.y_max, 1.2)
        assert_allclose(self.viewer.layers[0].mpl_hist, [0.5, 0.75, 1])
        assert_allclose(self.viewer.layers[0].mpl_bins, [-0.5, 0.5, 1.5, 2.5])
        assert_allclose(self.viewer.layers[1].mpl_hist, [0.5, 0.5, 1])
        assert_allclose(self.viewer.layers[1].mpl_bins, [-0.5, 0.5, 1.5, 2.5])

        viewer_state.normalize = False

        assert_allclose(self.viewer.state.y_max, 4.8)
        assert_allclose(self.viewer.layers[0].mpl_hist, [2, 3, 4])
        assert_allclose(self.viewer.layers[0].mpl_bins, [-0.5, 0.5, 1.5, 2.5])
        assert_allclose(self.viewer.layers[1].mpl_hist, [1, 1, 2])
        assert_allclose(self.viewer.layers[1].mpl_bins, [-0.5, 0.5, 1.5, 2.5])

        # TODO: add tests for log

    def test_apply_roi(self):

        # Check that when doing an ROI selection, the ROI clips to the bin edges
        # outside the selection

        viewer_state = self.viewer.state

        self.viewer.add_data(self.data)

        viewer_state.hist_x_min = -5
        viewer_state.hist_x_max = 5
        viewer_state.hist_n_bin = 4

        roi = XRangeROI(-0.2, 0.1)

        assert len(self.viewer.layers) == 1

        self.viewer.apply_roi(roi)

        assert len(self.viewer.layers) == 2

        assert_allclose(self.viewer.layers[0].mpl_hist, [0, 1, 2, 1])
        assert_allclose(self.viewer.layers[1].mpl_hist, [0, 1, 2, 0])

        assert_allclose(self.data.subsets[0].to_mask(), [0, 1, 1, 1])

        state = self.data.subsets[0].subset_state
        assert isinstance(state, RangeSubsetState)

        assert state.lo == -2.5
        assert state.hi == 2.5

        # TODO: add a similar test in log space

    def test_apply_roi_categorical(self):

        # Check that when doing an ROI selection, the ROI clips to the bin edges
        # outside the selection

        viewer_state = self.viewer.state

        self.viewer.add_data(self.data)

        viewer_state.x_att = self.data.id['y']

        roi = XRangeROI(0.3, 0.9)

        assert len(self.viewer.layers) == 1

        self.viewer.apply_roi(roi)

        assert len(self.viewer.layers) == 2

        assert_allclose(self.viewer.layers[0].mpl_hist, [2, 1, 1])
        assert_allclose(self.viewer.layers[1].mpl_hist, [2, 1, 0])

        assert_allclose(self.data.subsets[0].to_mask(), [1, 1, 0, 1])

        state = self.data.subsets[0].subset_state
        assert isinstance(state, CategoricalROISubsetState)

        assert_equal(state.roi.categories, ['a', 'b'])

    def test_apply_roi_empty(self):
        # Make sure that doing an ROI selection on an empty viewer doesn't
        # produce error messsages
        roi = XRangeROI(-0.2, 0.1)
        self.viewer.apply_roi(roi)

    def test_axes_labels(self):

        viewer_state = self.viewer.state

        self.viewer.add_data(self.data)

        assert self.viewer.axes.get_xlabel() == 'x'
        assert self.viewer.axes.get_ylabel() == 'Number'

        viewer_state.x_log = True

        assert self.viewer.axes.get_xlabel() == 'Log x'
        assert self.viewer.axes.get_ylabel() == 'Number'

        viewer_state.x_att = self.data.id['y']

        assert self.viewer.axes.get_xlabel() == 'y'
        assert self.viewer.axes.get_ylabel() == 'Number'

        viewer_state.normalize = True

        assert self.viewer.axes.get_xlabel() == 'y'
        assert self.viewer.axes.get_ylabel() == 'Normalized number'

        viewer_state.normalize = False
        viewer_state.cumulative = True

        assert self.viewer.axes.get_xlabel() == 'y'
        assert self.viewer.axes.get_ylabel() == 'Number'

    def test_y_min_y_max(self):

        # Regression test for a bug that caused y_max to not be set correctly
        # when multiple subsets were present and after turning on normalization
        # after switching to a different attribute from that used to make the
        # selection.

        viewer_state = self.viewer.state
        self.viewer.add_data(self.data)

        self.data.add_component([3.4, 3.5, 10.2, 20.3], 'z')

        viewer_state.x_att = self.data.id['x']

        cid = self.data.visible_components[0]
        self.data_collection.new_subset_group('subset 1', cid < 1)

        cid = self.data.visible_components[0]
        self.data_collection.new_subset_group('subset 2', cid < 2)

        cid = self.data.visible_components[0]
        self.data_collection.new_subset_group('subset 3', cid < 3)

        assert_allclose(self.viewer.state.y_min, 0)
        assert_allclose(self.viewer.state.y_max, 1.2)

        viewer_state.x_att = self.data.id['z']

        assert_allclose(self.viewer.state.y_min, 0)
        assert_allclose(self.viewer.state.y_max, 2.4)

        viewer_state.normalize = True

        assert_allclose(self.viewer.state.y_min, 0)
        assert_allclose(self.viewer.state.y_max, 0.5325443786982249)

    def test_update_when_limits_unchanged(self):

        # Regression test for glue-viz/glue#1010 - this bug caused histograms
        # to not be recomputed if the attribute changed but the limits and
        # number of bins did not.

        viewer_state = self.viewer.state

        self.viewer.add_data(self.data)

        viewer_state.x_att = self.data.id['y']
        viewer_state.hist_x_min = -10
        viewer_state.hist_x_max = +10
        viewer_state.hist_n_bin = 5

        assert_allclose(self.viewer.layers[0].mpl_hist, [0, 0, 3, 1, 0])

        viewer_state.x_att = self.data.id['x']
        viewer_state.hist_x_min = -10
        viewer_state.hist_x_max = +10
        viewer_state.hist_n_bin = 5

        assert_allclose(self.viewer.layers[0].mpl_hist, [0, 0, 2, 2, 0])

        viewer_state.x_att = self.data.id['y']

        assert_allclose(self.viewer.layers[0].mpl_hist, [0, 0, 3, 1, 0])

        viewer_state.x_att = self.data.id['x']

        assert_allclose(self.viewer.layers[0].mpl_hist, [0, 0, 2, 2, 0])

    def test_component_replaced(self):

        # regression test for 508 - if a component ID is replaced, we should
        # make sure that the component ID is selected if the old component ID
        # was selected

        self.viewer.add_data(self.data)
        self.viewer.state.x_att = self.data.id['x']
        test = ComponentID('test')
        self.data.update_id(self.viewer.state.x_att, test)
        assert self.viewer.state.x_att is test
        assert combo_as_string(self.viewer.options_widget().ui.combosel_x_att) == 'Main components:test:y:Coordinate components:Pixel Axis 0 [x]:World 0'

    def test_nbin_override_persists_over_numerical_attribute_change(self):

        # regression test for #398

        self.data.add_component([3, 4, 1, 2], 'z')

        self.viewer.add_data(self.data)
        self.viewer.state.x_att = self.data.id['x']
        self.viewer.state.hist_n_bin = 7
        self.viewer.state.x_att = self.data.id['z']
        assert self.viewer.state.hist_n_bin == 7

    @pytest.mark.parametrize('protocol', [0, 1])
    def test_session_back_compat(self, protocol):

        filename = os.path.join(DATA, 'histogram_v{0}.glu'.format(protocol))

        with open(filename, 'r') as f:
            session = f.read()

        state = GlueUnSerializer.loads(session)

        ga = state.object('__main__')

        dc = ga.session.data_collection

        assert len(dc) == 1

        assert dc[0].label == 'data'

        viewer1 = ga.viewers[0][0]
        assert len(viewer1.state.layers) == 2
        assert viewer1.state.x_att is dc[0].id['a']
        assert_allclose(viewer1.state.x_min, 0)
        assert_allclose(viewer1.state.x_max, 9)
        assert_allclose(viewer1.state.y_min, 0)
        assert_allclose(viewer1.state.y_max, 2.4)
        assert_allclose(viewer1.state.hist_x_min, 0)
        assert_allclose(viewer1.state.hist_x_max, 9)
        assert_allclose(viewer1.state.hist_n_bin, 6)
        assert not viewer1.state.x_log
        assert not viewer1.state.y_log
        assert viewer1.state.layers[0].visible
        assert not viewer1.state.layers[1].visible
        assert not viewer1.state.cumulative
        assert not viewer1.state.normalize

        viewer2 = ga.viewers[0][1]
        assert viewer2.state.x_att is dc[0].id['b']
        assert_allclose(viewer2.state.x_min, 2)
        assert_allclose(viewer2.state.x_max, 16)
        assert_allclose(viewer2.state.y_min, 0)
        assert_allclose(viewer2.state.y_max, 1.2)
        assert_allclose(viewer2.state.hist_x_min, 2)
        assert_allclose(viewer2.state.hist_x_max, 16)
        assert_allclose(viewer2.state.hist_n_bin, 8)
        assert not viewer2.state.x_log
        assert not viewer2.state.y_log
        assert viewer2.state.layers[0].visible
        assert viewer2.state.layers[1].visible
        assert not viewer2.state.cumulative
        assert not viewer2.state.normalize

        viewer3 = ga.viewers[0][2]
        assert viewer3.state.x_att is dc[0].id['a']
        assert_allclose(viewer3.state.x_min, 0)
        assert_allclose(viewer3.state.x_max, 9)
        assert_allclose(viewer3.state.y_min, 0.01111111111111111)
        assert_allclose(viewer3.state.y_max, 0.7407407407407407)
        assert_allclose(viewer3.state.hist_x_min, 0)
        assert_allclose(viewer3.state.hist_x_max, 9)
        assert_allclose(viewer3.state.hist_n_bin, 10)
        assert not viewer3.state.x_log
        assert viewer3.state.y_log
        assert viewer3.state.layers[0].visible
        assert viewer3.state.layers[1].visible
        assert not viewer3.state.cumulative
        assert viewer3.state.normalize

        viewer4 = ga.viewers[0][3]
        assert viewer4.state.x_att is dc[0].id['a']
        assert_allclose(viewer4.state.x_min, -1)
        assert_allclose(viewer4.state.x_max, 10)
        assert_allclose(viewer4.state.y_min, 0)
        assert_allclose(viewer4.state.y_max, 12)
        assert_allclose(viewer4.state.hist_x_min, -1)
        assert_allclose(viewer4.state.hist_x_max, 10)
        assert_allclose(viewer4.state.hist_n_bin, 4)
        assert not viewer4.state.x_log
        assert not viewer4.state.y_log
        assert viewer4.state.layers[0].visible
        assert viewer4.state.layers[1].visible
        assert viewer4.state.cumulative
        assert not viewer4.state.normalize

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

        d1 = Data(a=[1, 2, 3], label='d1')
        d2 = Data(b=[1, 2, 3], label='d2')
        d3 = Data(c=[1, 2, 3], label='d3')
        d4 = Data(d=[1, 2, 3], label='d4')

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

    def test_datetime64_support(self, tmpdir):

        self.data.add_component(np.array([100, 200, 300, 400], dtype='M8[D]'), 't1')
        self.viewer.add_data(self.data)
        self.viewer.state.x_att = self.data.id['t1']

        # Matplotlib deals with dates by converting them to the number of days
        # since 01-01-0001, so we can check that the limits are correctly
        # converted (and not 100 to 400)
        assert self.viewer.axes.get_xlim() == (719263.0, 719563.0)

        # Apply an ROI selection in plotting coordinates
        roi = XRangeROI(719313, 719513)
        self.viewer.apply_roi(roi)

        # Check that the two middle elements are selected
        assert_equal(self.data.subsets[0].to_mask(), [0, 1, 1, 0])

        # Make sure that the Qt labels look ok
        options = self.viewer.options_widget().ui
        assert options.valuetext_x_min.text() == '1970-04-11'
        assert options.valuetext_x_max.text() == '1971-02-05'

        # Make sure that we can set the xmin/xmax to a string date
        assert_equal(self.viewer.state.x_min, np.datetime64('1970-04-11', 'D'))
        options.valuetext_x_min.setText('1970-04-14')
        options.valuetext_x_min.editingFinished.emit()
        assert self.viewer.axes.get_xlim() == (719266.0, 719563.0)
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
        assert options.valuetext_x_max.text() == '1971-02-05'
