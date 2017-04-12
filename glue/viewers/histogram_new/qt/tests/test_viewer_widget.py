# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import os

import pytest

from numpy.testing import assert_equal, assert_allclose

from glue.core import Data
from glue.core.roi import XRangeROI
from glue.core.subset import RangeSubsetState, CategoricalROISubsetState
from glue import core
from glue.core.component_id import ComponentID
from glue.core.tests.util import simple_session
from glue.utils.qt import combo_as_string
from glue.viewers.common.qt.tests.test_mpl_data_viewer import BaseTestMatplotlibDataViewer
from glue.core.state import GlueUnSerializer

from ..data_viewer import HistogramViewer

DATA = os.path.join(os.path.dirname(__file__), 'data')


class TestHistogramCommon(BaseTestMatplotlibDataViewer):
    def init_data(self):
        return Data(label='d1', x=[3.4, 2.3, -1.1, 0.3], y=['a', 'b', 'c', 'a'])
    viewer_cls = HistogramViewer


class TestHistogramViewer(object):

    def setup_method(self, method):

        self.data = Data(label='d1', x=[3.4, 2.3, -1.1, 0.3], y=['a', 'b', 'c', 'a'])

        self.session = simple_session()
        self.hub = self.session.hub

        self.data_collection = self.session.data_collection
        self.data_collection.append(self.data)

        self.viewer = HistogramViewer(self.session)

        self.data_collection.register_to_hub(self.hub)
        self.viewer.register_to_hub(self.hub)

    def teardown_method(self, method):
        self.viewer.close()

    def test_basic(self):

        viewer_state = self.viewer.state

        # Check defaults when we add data
        self.viewer.add_data(self.data)

        assert combo_as_string(self.viewer.options_widget().ui.combodata_xatt) == 'x:y'

        assert viewer_state.xatt is self.data.id['x']
        assert viewer_state.x_min == -1.1
        assert viewer_state.x_max == 3.4
        assert viewer_state.y_min == 0.0
        assert viewer_state.y_max == 1.2

        assert viewer_state.hist_x_min == -1.1
        assert viewer_state.hist_x_max == 3.4
        assert viewer_state.hist_n_bin == 15

        assert not viewer_state.cumulative
        assert not viewer_state.normalize

        assert not viewer_state.log_x
        assert not viewer_state.log_y

        assert len(viewer_state.layers) == 1

        # Change to categorical component and check new values

        viewer_state.xatt = self.data.id['y']

        assert viewer_state.x_min == -0.5
        assert viewer_state.x_max == 2.5
        assert viewer_state.y_min == 0.0
        assert viewer_state.y_max == 2.4

        assert viewer_state.hist_x_min == -0.5
        assert viewer_state.hist_x_max == 2.5
        assert viewer_state.hist_n_bin == 3

        assert not viewer_state.cumulative
        assert not viewer_state.normalize

        assert not viewer_state.log_x
        assert not viewer_state.log_y

    def test_remove_data(self):
        self.viewer.add_data(self.data)
        assert combo_as_string(self.viewer.options_widget().ui.combodata_xatt) == 'x:y'
        self.data_collection.remove(self.data)
        assert combo_as_string(self.viewer.options_widget().ui.combodata_xatt) == ''

    def test_update_component_updates_title(self):
        self.viewer.add_data(self.data)
        self.viewer.windowTitle() == 'x'
        self.viewer.state.xatt = self.data.id['y']
        self.viewer.windowTitle() == 'y'

    def test_combo_updates_with_component_add(self):
        self.viewer.add_data(self.data)
        self.data.add_component([3, 4, 1, 2], 'z')
        assert self.viewer.state.xatt is self.data.id['x']
        assert combo_as_string(self.viewer.options_widget().ui.combodata_xatt) == 'x:y:z'

    def test_nonnumeric_first_component(self):
        # regression test for #208. Shouldn't complain if
        # first component is non-numerical
        data = core.Data()
        data.add_component(['a', 'b', 'c'], label='c1')
        data.add_component([1, 2, 3], label='c2')
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

        assert_allclose(self.viewer.layers[0].mpl_hist, [0, 1, 2, 1])
        assert_allclose(self.viewer.layers[0].mpl_bins, [-5, -2.5, 0, 2.5, 5])

        cid = self.data.visible_components[0]
        self.data_collection.new_subset_group('subset 1', cid < 2)

        assert_allclose(self.viewer.layers[1].mpl_hist, [0, 1, 1, 0])
        assert_allclose(self.viewer.layers[1].mpl_bins, [-5, -2.5, 0, 2.5, 5])

        viewer_state.normalize = True

        assert_allclose(self.viewer.layers[0].mpl_hist, [0, 0.1, 0.2, 0.1])
        assert_allclose(self.viewer.layers[0].mpl_bins, [-5, -2.5, 0, 2.5, 5])
        assert_allclose(self.viewer.layers[1].mpl_hist, [0, 0.2, 0.2, 0])
        assert_allclose(self.viewer.layers[1].mpl_bins, [-5, -2.5, 0, 2.5, 5])

        viewer_state.cumulative = True

        assert_allclose(self.viewer.layers[0].mpl_hist, [0, 0.25, 0.75, 1.0])
        assert_allclose(self.viewer.layers[0].mpl_bins, [-5, -2.5, 0, 2.5, 5])
        assert_allclose(self.viewer.layers[1].mpl_hist, [0, 0.5, 1.0, 1.0])
        assert_allclose(self.viewer.layers[1].mpl_bins, [-5, -2.5, 0, 2.5, 5])

        viewer_state.normalize = False

        assert_allclose(self.viewer.layers[0].mpl_hist, [0, 1, 3, 4])
        assert_allclose(self.viewer.layers[0].mpl_bins, [-5, -2.5, 0, 2.5, 5])
        assert_allclose(self.viewer.layers[1].mpl_hist, [0, 1, 2, 2])
        assert_allclose(self.viewer.layers[1].mpl_bins, [-5, -2.5, 0, 2.5, 5])

        viewer_state.cumulative = False

        # Categorical attribute

        viewer_state.xatt = self.data.id['y']

        formatter = self.viewer.axes.xaxis.get_major_formatter()
        xlabels = [formatter.format_data(pos) for pos in range(3)]
        assert xlabels == ['a', 'b', 'c']

        assert_allclose(self.viewer.layers[0].mpl_hist, [2, 1, 1])
        assert_allclose(self.viewer.layers[0].mpl_bins, [-0.5, 0.5, 1.5, 2.5])
        assert_allclose(self.viewer.layers[1].mpl_hist, [1, 0, 1])
        assert_allclose(self.viewer.layers[1].mpl_bins, [-0.5, 0.5, 1.5, 2.5])

        viewer_state.normalize = True

        assert_allclose(self.viewer.layers[0].mpl_hist, [0.5, 0.25, 0.25])
        assert_allclose(self.viewer.layers[0].mpl_bins, [-0.5, 0.5, 1.5, 2.5])
        assert_allclose(self.viewer.layers[1].mpl_hist, [0.5, 0, 0.5])
        assert_allclose(self.viewer.layers[1].mpl_bins, [-0.5, 0.5, 1.5, 2.5])

        viewer_state.cumulative = True

        assert_allclose(self.viewer.layers[0].mpl_hist, [0.5, 0.75, 1])
        assert_allclose(self.viewer.layers[0].mpl_bins, [-0.5, 0.5, 1.5, 2.5])
        assert_allclose(self.viewer.layers[1].mpl_hist, [0.5, 0.5, 1])
        assert_allclose(self.viewer.layers[1].mpl_bins, [-0.5, 0.5, 1.5, 2.5])

        viewer_state.normalize = False

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

        viewer_state.xatt = self.data.id['y']

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

    def test_axes_labels(self):

        viewer_state = self.viewer.state

        self.viewer.add_data(self.data)

        assert self.viewer.axes.get_xlabel() == 'x'
        assert self.viewer.axes.get_ylabel() == 'Number'

        viewer_state.log_x = True

        assert self.viewer.axes.get_xlabel() == 'Log x'
        assert self.viewer.axes.get_ylabel() == 'Number'

        viewer_state.xatt = self.data.id['y']

        assert self.viewer.axes.get_xlabel() == 'y'
        assert self.viewer.axes.get_ylabel() == 'Number'

        viewer_state.normalize = True

        assert self.viewer.axes.get_xlabel() == 'y'
        assert self.viewer.axes.get_ylabel() == 'Normalized number'

        viewer_state.normalize = False
        viewer_state.cumulative = True

        assert self.viewer.axes.get_xlabel() == 'y'
        assert self.viewer.axes.get_ylabel() == 'Number'

    def test_update_when_limits_unchanged(self):

        # Regression test for glue-viz/glue#1010 - this bug caused histograms
        # to not be recomputed if the attribute changed but the limits and
        # number of bins did not.

        viewer_state = self.viewer.state

        self.viewer.add_data(self.data)

        viewer_state.xatt = self.data.id['y']
        viewer_state.hist_x_min = -10
        viewer_state.hist_x_max = +10
        viewer_state.hist_n_bin = 5

        assert_allclose(self.viewer.layers[0].mpl_hist, [0, 0, 3, 1, 0])

        viewer_state.xatt = self.data.id['x']
        viewer_state.hist_x_min = -10
        viewer_state.hist_x_max = +10
        viewer_state.hist_n_bin = 5

        assert_allclose(self.viewer.layers[0].mpl_hist, [0, 0, 2, 2, 0])

        viewer_state.xatt = self.data.id['y']

        assert_allclose(self.viewer.layers[0].mpl_hist, [0, 0, 3, 1, 0])

        viewer_state.xatt = self.data.id['x']

        assert_allclose(self.viewer.layers[0].mpl_hist, [0, 0, 2, 2, 0])

    # TODO: update the following test following refactoring
    def test_component_replaced(self):

        # regression test for 508 - if a component ID is replaced, we should
        # make sure that the component ID is selected if the old component ID
        # was selected

        self.viewer.add_data(self.data)
        self.viewer.state.xatt = self.data.components[0]
        test = ComponentID('test')
        self.data.update_id(self.viewer.state.xatt, test)
        assert self.viewer.state.xatt is test
        assert combo_as_string(self.viewer.options_widget().ui.combodata_xatt) == 'test:y'

    # TODO: Check for extraneous draw events

    @pytest.mark.xfail
    def test_nbin_override_persists_over_attribute_change(self):

        # regression test for #398

        self.viewer.add_data(self.data)
        self.viewer.state.xatt = self.data.id['x']
        self.viewer.state.hist_n_bin = 7
        self.viewer.state.xatt = self.data.id['y']
        assert self.viewer.state.hist_n_bin == 7

    @pytest.mark.parametrize('protocol', [0])
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
        assert viewer1.state.xatt is dc[0].id['a']
        assert_allclose(viewer1.state.x_min, 0)
        assert_allclose(viewer1.state.x_max, 9)
        assert_allclose(viewer1.state.y_min, 0)
        assert_allclose(viewer1.state.y_max, 2.4)
        assert_allclose(viewer1.state.hist_x_min, 0)
        assert_allclose(viewer1.state.hist_x_max, 9)
        assert_allclose(viewer1.state.hist_n_bin, 6)
        assert not viewer1.state.log_x
        assert not viewer1.state.log_y
        assert viewer1.state.layers[0].visible
        assert not viewer1.state.layers[1].visible
        assert not viewer1.state.cumulative
        assert not viewer1.state.normalize

        viewer2 = ga.viewers[0][1]
        assert viewer2.state.xatt is dc[0].id['b']
        assert_allclose(viewer2.state.x_min, 2)
        assert_allclose(viewer2.state.x_max, 16)
        assert_allclose(viewer2.state.y_min, 0)
        assert_allclose(viewer2.state.y_max, 1.2)
        assert_allclose(viewer2.state.hist_x_min, 2)
        assert_allclose(viewer2.state.hist_x_max, 16)
        assert_allclose(viewer2.state.hist_n_bin, 8)
        assert not viewer2.state.log_x
        assert not viewer2.state.log_y
        assert viewer2.state.layers[0].visible
        assert viewer2.state.layers[1].visible
        assert not viewer2.state.cumulative
        assert not viewer2.state.normalize

        viewer3 = ga.viewers[0][2]
        assert viewer3.state.xatt is dc[0].id['a']
        assert_allclose(viewer3.state.x_min, 0)
        assert_allclose(viewer3.state.x_max, 9)
        assert_allclose(viewer3.state.y_min, 0.037037037037037035)
        assert_allclose(viewer3.state.y_max, 0.7407407407407407)
        assert_allclose(viewer3.state.hist_x_min, 0)
        assert_allclose(viewer3.state.hist_x_max, 9)
        assert_allclose(viewer3.state.hist_n_bin, 10)
        assert not viewer3.state.log_x
        assert viewer3.state.log_y
        assert viewer3.state.layers[0].visible
        assert viewer3.state.layers[1].visible
        assert not viewer3.state.cumulative
        assert viewer3.state.normalize

        viewer4 = ga.viewers[0][3]
        assert viewer4.state.xatt is dc[0].id['a']
        assert_allclose(viewer4.state.x_min, -1)
        assert_allclose(viewer4.state.x_max, 10)
        assert_allclose(viewer4.state.y_min, 0)
        assert_allclose(viewer4.state.y_max, 12)
        assert_allclose(viewer4.state.hist_x_min, -1)
        assert_allclose(viewer4.state.hist_x_max, 10)
        assert_allclose(viewer4.state.hist_n_bin, 4)
        assert not viewer4.state.log_x
        assert not viewer4.state.log_y
        assert viewer4.state.layers[0].visible
        assert viewer4.state.layers[1].visible
        assert viewer4.state.cumulative
        assert not viewer4.state.normalize
