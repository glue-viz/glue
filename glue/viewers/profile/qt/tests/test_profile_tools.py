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
from glue.utils.qt import combo_as_string, get_qapp
from glue.viewers.matplotlib.qt.tests.test_data_viewer import BaseTestMatplotlibDataViewer
from glue.core.state import GlueUnSerializer
from glue.app.qt.layer_tree_widget import LayerTreeWidget

from glue.viewers.image.qt import ImageViewer
from ..data_viewer import ProfileViewer

class TestProfileTools(object):

    def setup_method(self, method):

        self.data = Data(label='d1', x=np.arange(240).reshape((30, 4, 2)))

        self.app = GlueApplication()
        self.session = self.app.session
        self.hub = self.session.hub

        self.data_collection = self.session.data_collection
        self.data_collection.append(self.data)

        self.viewer = self.app.new_data_viewer(ProfileViewer)

        self.viewer.toolbar.active_tool = 'profile-analysis'

        self.profile_tools = self.viewer.toolbar.tools['profile-analysis']._profile_tools

    def teardown_method(self, method):
        self.viewer.close()

    def test_navigate_sync_image(self):
        self.viewer.add_data(self.data)
        image_viewer = self.app.new_data_viewer(ImageViewer)
        image_viewer.add_data(self.data)
        assert image_viewer.state.slices == (0, 0, 0)
        x, y = self.viewer.axes.transData.transform([[1, 4]])[0]
        self.viewer.axes.figure.canvas.button_press_event(x, y, 1)
        assert image_viewer.state.slices == (1, 0, 0)

    def test_fit_polynomial(self):
        # TODO: need to deterministically set to polynomial fitter
        self.viewer.add_data(self.data)
        self.profile_tools.ui.tabs.setCurrentIndex(1)
        x, y = self.viewer.axes.transData.transform([[1, 4]])[0]
        self.viewer.axes.figure.canvas.button_press_event(x, y, 1)
        x, y = self.viewer.axes.transData.transform([[15, 4]])[0]
        self.viewer.axes.figure.canvas.motion_notify_event(x, y, 1)
        assert_allclose(self.profile_tools.rng_mode.state.x_range, (1, 15))
        self.profile_tools.ui.button_fit.click()
        self.profile_tools.wait_for_fit()
        app = get_qapp()
        app.processEvents()
        assert self.profile_tools.text_log.toPlainText().startswith('d1\nCoefficients')
