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

from ..data_viewer import ProfileViewer

DATA = os.path.join(os.path.dirname(__file__), 'data')


class TestProfileCommon(BaseTestMatplotlibDataViewer):
    def init_data(self):
        return Data(label='d1',
                    x=np.random.random(24).reshape((3, 4, 2)))
    viewer_cls = ProfileViewer


class TestProfileViewer(object):

    def setup_method(self, method):

        self.data = Data(label='d1', x=np.arange(24).reshape((3, 4, 2)))

        self.app = GlueApplication()
        self.session = self.app.session
        self.hub = self.session.hub

        self.data_collection = self.session.data_collection
        self.data_collection.append(self.data)

        self.viewer = self.app.new_data_viewer(ProfileViewer)

    def teardown_method(self, method):
        self.viewer.close()

    def test_functions(self):
        self.viewer.add_data(self.data)
        self.viewer.state.function = np.nanmean
        assert len(self.viewer.layers) == 1
        layer_artist = self.viewer.layers[0]
        assert_allclose(layer_artist._visible_data[0], [0, 1, 2])
        assert_allclose(layer_artist._visible_data[1], [3.5, 11.5, 19.5])
