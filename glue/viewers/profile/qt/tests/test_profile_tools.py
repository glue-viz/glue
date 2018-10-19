from __future__ import absolute_import, division, print_function

import pytest
import numpy as np

from numpy.testing import assert_allclose

from glue.core import Data
from glue.tests.helpers import PYSIDE2_INSTALLED, MATPLOTLIB_GE_30_INSTALLED  # noqa
from glue.app.qt import GlueApplication
from glue.utils import nanmean
from glue.utils.qt import get_qapp
from glue.viewers.image.state import AggregateSlice

from glue.viewers.image.qt import ImageViewer
from glue.viewers.profile.tests.test_state import SimpleCoordinates
from ..data_viewer import ProfileViewer


class TestProfileTools(object):

    def setup_method(self, method):

        self.data = Data(label='d1')
        self.data.coords = SimpleCoordinates()
        self.data['x'] = np.arange(240).reshape((30, 4, 2)).astype(float)

        self.app = GlueApplication()
        self.session = self.app.session
        self.hub = self.session.hub

        self.data_collection = self.session.data_collection
        self.data_collection.append(self.data)

        self.viewer = self.app.new_data_viewer(ProfileViewer)
        self.viewer.state.function = nanmean

        self.viewer.toolbar.active_tool = 'profile-analysis'

        self.profile_tools = self.viewer.toolbar.tools['profile-analysis']._profile_tools

    def teardown_method(self, method):
        self.viewer.close()
        self.viewer = None
        self.app.close()
        self.app = None

    @pytest.mark.skipif('MATPLOTLIB_GE_30_INSTALLED')
    def test_navigate_sync_image(self):

        self.viewer.add_data(self.data)
        image_viewer = self.app.new_data_viewer(ImageViewer)
        image_viewer.add_data(self.data)
        assert image_viewer.state.slices == (0, 0, 0)

        self.viewer.state.x_att = self.data.pixel_component_ids[0]

        x, y = self.viewer.axes.transData.transform([[1, 4]])[0]
        self.viewer.axes.figure.canvas.button_press_event(x, y, 1)
        self.viewer.axes.figure.canvas.button_release_event(x, y, 1)
        assert image_viewer.state.slices == (1, 0, 0)

        self.viewer.state.x_att = self.data.world_component_ids[0]

        x, y = self.viewer.axes.transData.transform([[10, 4]])[0]
        self.viewer.axes.figure.canvas.button_press_event(x, y, 1)
        self.viewer.axes.figure.canvas.button_release_event(x, y, 1)
        assert image_viewer.state.slices == (5, 0, 0)

    @pytest.mark.skipif('PYSIDE2_INSTALLED or MATPLOTLIB_GE_30_INSTALLED')
    def test_fit_polynomial(self):

        # TODO: need to deterministically set to polynomial fitter

        self.viewer.add_data(self.data)
        self.profile_tools.ui.tabs.setCurrentIndex(1)

        # First try in pixel coordinates

        self.viewer.state.x_att = self.data.pixel_component_ids[0]

        x, y = self.viewer.axes.transData.transform([[0.9, 4]])[0]
        self.viewer.axes.figure.canvas.button_press_event(x, y, 1)
        x, y = self.viewer.axes.transData.transform([[15.1, 4]])[0]
        self.viewer.axes.figure.canvas.motion_notify_event(x, y, 1)

        assert_allclose(self.profile_tools.rng_mode.state.x_range, (0.9, 15.1))

        self.profile_tools.ui.button_fit.click()
        self.profile_tools.wait_for_fit()
        app = get_qapp()
        app.processEvents()

        pixel_log = self.profile_tools.text_log.toPlainText().splitlines()
        assert pixel_log[0] == 'd1'
        assert pixel_log[1] == 'Coefficients:'
        assert pixel_log[-2] == '8.000000e+00'
        assert pixel_log[-1] == '3.500000e+00'

        self.profile_tools.ui.button_clear.click()
        assert self.profile_tools.text_log.toPlainText() == ''

        # Next, try in world coordinates

        self.viewer.state.x_att = self.data.world_component_ids[0]

        x, y = self.viewer.axes.transData.transform([[1.9, 4]])[0]
        self.viewer.axes.figure.canvas.button_press_event(x, y, 1)
        x, y = self.viewer.axes.transData.transform([[30.1, 4]])[0]
        self.viewer.axes.figure.canvas.motion_notify_event(x, y, 1)

        assert_allclose(self.profile_tools.rng_mode.state.x_range, (1.9, 30.1))

        self.profile_tools.ui.button_fit.click()
        self.profile_tools.wait_for_fit()
        app = get_qapp()
        app.processEvents()

        world_log = self.profile_tools.text_log.toPlainText().splitlines()
        assert world_log[0] == 'd1'
        assert world_log[1] == 'Coefficients:'
        assert world_log[-2] == '4.000000e+00'
        assert world_log[-1] == '3.500000e+00'

    @pytest.mark.skipif('MATPLOTLIB_GE_30_INSTALLED')
    def test_collapse(self):

        self.viewer.add_data(self.data)

        image_viewer = self.app.new_data_viewer(ImageViewer)
        image_viewer.add_data(self.data)

        self.profile_tools.ui.tabs.setCurrentIndex(2)

        # First try in pixel coordinates

        self.viewer.state.x_att = self.data.pixel_component_ids[0]

        x, y = self.viewer.axes.transData.transform([[0.9, 4]])[0]
        self.viewer.axes.figure.canvas.button_press_event(x, y, 1)
        x, y = self.viewer.axes.transData.transform([[15.1, 4]])[0]
        self.viewer.axes.figure.canvas.motion_notify_event(x, y, 1)

        self.profile_tools.ui.button_collapse.click()

        assert isinstance(image_viewer.state.slices[0], AggregateSlice)
        assert image_viewer.state.slices[0].slice.start == 1
        assert image_viewer.state.slices[0].slice.stop == 15
        assert image_viewer.state.slices[0].center == 0
        assert image_viewer.state.slices[0].function is nanmean

        # Next, try in world coordinates

        self.viewer.state.x_att = self.data.world_component_ids[0]

        x, y = self.viewer.axes.transData.transform([[1.9, 4]])[0]
        self.viewer.axes.figure.canvas.button_press_event(x, y, 1)
        x, y = self.viewer.axes.transData.transform([[30.1, 4]])[0]
        self.viewer.axes.figure.canvas.motion_notify_event(x, y, 1)

        self.profile_tools.ui.button_collapse.click()

        assert isinstance(image_viewer.state.slices[0], AggregateSlice)
        assert image_viewer.state.slices[0].slice.start == 1
        assert image_viewer.state.slices[0].slice.stop == 15
        assert image_viewer.state.slices[0].center == 0
        assert image_viewer.state.slices[0].function is nanmean
