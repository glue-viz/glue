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
