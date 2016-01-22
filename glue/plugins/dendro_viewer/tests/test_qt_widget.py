# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
import pytest

from glue.qt.widgets.tests import simple_session
from glue import core

from ..qt_widget import DendroWidget
from glue.qt.widgets.tests.test_data_viewer import BaseTestDataViewer

def mock_data():
    return core.Data(label='d1', x=[1, 2, 3], y=[2, 3, 4])

import os
os.environ['GLUE_TESTING'] = 'True'


class TestDendroWidget(object):

    def setup_method(self, method):
        s = simple_session()
        self.hub = s.hub
        self.data = core.Data(label='d1', x=[1, 2, 3])
        self.dc = s.data_collection
        self.dc.append(self.data)

        self.w = DendroWidget(s)

    def test_ignore_double_add(self):
        self.w.add_data(self.data)
        assert self.data in self.w.client
        self.w.add_data(self.data)

    def test_update_combos_empty_data(self):
        self.w._update_combos()

    def test_add_subset(self):
        s = self.data.new_subset()
        self.w.add_subset(s)
        assert self.data in self.w.client
        assert s in self.w.client


class TestDataViewerDendro(BaseTestDataViewer):
    # A few additional tests common to all data viewers
    widget_cls = DendroWidget
