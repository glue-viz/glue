from __future__ import absolute_import, division, print_function

from mock import MagicMock

from ..application_base import Application
from .. import Data
from ...external.six.moves import cPickle as pickle


class MockApplication(Application):

    def __init__(self, data_collection=None, session=None):
        super(MockApplication, self).__init__(data_collection=data_collection, session=session)
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

    def test_suggest_mergers(self):
        x = Data(x=[1, 2, 3])
        y = Data(y=[1, 2, 3, 4])
        z = Data(z=[1, 2, 3])

        Application._choose_merge = MagicMock()
        Application._choose_merge.return_value = [x]
        self.app.data_collection.merge = MagicMock()

        self.app.data_collection.append(x)
        self.app.data_collection.append(y)

        self.app.add_datasets(self.app.data_collection, z)

        args = self.app._choose_merge.call_args[0]
        assert args[0] == z
        assert args[1] == [x]

        assert self.app.data_collection.merge.call_count == 1


def test_session(tmpdir):
    session_file = tmpdir.join('test.glu').strpath
    app = MockApplication()
    app.save_session(session_file)
    app2 = MockApplication.restore_session(session_file)
