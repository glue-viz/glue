from cPickle import load
import tempfile
import os

import numpy as np
from mock import MagicMock

from ..application_base import Application
from .. import Data


class MockApplication(Application):

    def __init__(self, data=None, hub=None):
        super(MockApplication, self).__init__(data, hub)
        self.tab = MagicMock()
        self.errors = MagicMock()

    def report_error(self, message, detail):
        self.errors.report(message, detail)

    def new_tab(self):
        self.tab.tab()

    def add_widget(self, widget, label=None, tab=None):
        self.tab.add(widget, label)

    def close_tab(self):
        self.tab.close()

    def _load_settings(self):
        pass


class TestApplicationBase(object):
    def setup_method(self, method):
        self.app = MockApplication()

    def test_save_session(self):
        self.app._data.append(Data(label='x', x=[1, 2, 3]))
        _, fname = tempfile.mkstemp(suffix='.glu')
        self.app.save_session(fname)

        dc, hub = load(open(fname))
        data = dc[0]
        np.testing.assert_array_equal(data['x'], [1, 2, 3])

        os.unlink(fname)
