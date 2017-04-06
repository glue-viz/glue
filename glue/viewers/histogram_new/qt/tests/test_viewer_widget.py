# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function


from glue.core import Data
from glue import core
from glue.core.tests.util import simple_session
from glue.utils.qt import combo_as_string
from glue.viewers.common.qt.tests.test_mpl_data_viewer import BaseTestMatplotlibDataViewer

from ..data_viewer import HistogramViewer


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

        viewer_state = self.viewer.viewer_state

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
        self.viewer.viewer_state.xatt = self.data.id['y']
        self.viewer.windowTitle() == 'y'

    def test_combo_updates_with_component_add(self):
        self.viewer.add_data(self.data)
        self.data.add_component([3, 4, 1, 2], 'z')
        assert self.viewer.viewer_state.xatt is self.data.id['x']
        assert combo_as_string(self.viewer.options_widget().ui.combodata_xatt) == 'x:y:z'

    def test_nonnumeric_first_component(self):
        # regression test for #208. Shouldn't complain if
        # first component is non-numerical
        data = core.Data()
        data.add_component(['a', 'b', 'c'], label='c1')
        data.add_component([1, 2, 3], label='c2')
        self.data_collection.append(data)
        self.viewer.add_data(data)
