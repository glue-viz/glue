from __future__ import absolute_import, division, print_function

import os
import json
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


def test_session_paths(tmpdir):

    os.chdir(tmpdir.strpath)
    tmpdir.mkdir('data')

    with open(os.path.join('data', 'data.csv'), 'w') as f:
        f.write('a, b\n1, 2\n3, 4\n')

    app = MockApplication()
    app.load_data(os.path.join('data', 'data.csv'))

    # Make sure the paths work whether the session file is saved in the current
    # directory or in a sub-directory (this is important for relative links)
    for save_dir in ('.', 'data'):

        for absolute in (True, False):

            print(save_dir, absolute)

            session_file = tmpdir.join(save_dir).join('test.glu').strpath

            app.save_session(session_file, absolute_paths=absolute)

            with open(session_file) as f:
                data = json.load(f)

            for key, value in data.items():
                if value.get('_type') == 'glue.core.data_factories.helpers.LoadLog':
                    assert os.path.isabs(value['path']) is absolute

            MockApplication.restore_session(session_file)


def test_load_data(tmpdir):

    os.chdir(tmpdir.strpath)

    with open('data1.csv', 'w') as f:
        f.write('a, b\n1, 2\n3, 4\n')

    with open('data2.csv', 'w') as f:
        f.write('a, b\n1, 2\n3, 4\n')

    app1 = Application()
    data = app1.load_data('data1.csv')

    assert len(app1.data_collection) == 1
    assert isinstance(data, Data)

    app2 = Application()
    datasets = app2.load_data(['data1.csv', 'data2.csv'], skip_merge=True)

    assert len(app2.data_collection) == 2
    assert len(datasets) == 2
    assert isinstance(datasets[0], Data)
    assert isinstance(datasets[1], Data)

    app3 = Application()
    data = app3.load_data(['data1.csv', 'data2.csv'], auto_merge=True)

    assert len(app3.data_collection) == 1
    # NOTE: for now this still returns the individual datasets
    assert len(datasets) == 2
    assert isinstance(datasets[0], Data)
    assert isinstance(datasets[1], Data)
