from __future__ import absolute_import, division, print_function

from mock import MagicMock

from .. import Data
from ..application_base import Application


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
        Application._choose_merge.return_value = ([x], 'mydata')
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
    MockApplication.restore_session(session_file)


def test_set_data_color():

    x = Data(x=[1, 2, 3])
    y = Data(y=[1, 2, 3, 4])

    x.style.color = 'blue'
    x.style.alpha = 0.4
    y.style.color = 'purple'
    y.style.alpha = 0.5

    app = Application()
    app.data_collection.append(x)
    app.data_collection.append(y)

    app.set_data_color('red', alpha=0.3)

    assert x.style.color == 'red'
    assert x.style.alpha == 0.3

    assert y.style.color == 'red'
    assert y.style.alpha == 0.3


def test_nested_data():

    # Regression test that caused add_datasets to crash if datasets included
    # lists of lists of data, which is possible if a data factory returns more
    # than one data object.

    x = Data(x=[1, 2])
    y = Data(x=[1, 2, 3])
    z = Data(y=[1, 2, 3, 4])
    a = Data(y=[1, 2, 3, 4, 5])
    b = Data(y=[1, 2, 3, 4, 5, 6])
    c = Data(y=[1, 2, 3, 4, 5, 6, 7])

    datasets = [x, [[[y, z]], a], [[[[b, c]]]]]


    app = Application()

    app.add_datasets(app.data_collection, datasets)

    assert x in app.data_collection
    assert y in app.data_collection
    assert z in app.data_collection
    assert a in app.data_collection
    assert b in app.data_collection
    assert c in app.data_collection
