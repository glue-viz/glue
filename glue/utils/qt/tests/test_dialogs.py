from __future__ import absolute_import, division, print_function

import mock

from ..dialogs import pick_item, pick_class, get_text


def test_pick_item():

    items = ['a', 'b', 'c']
    labels = ['1', '2', '3']

    with mock.patch('qtpy.QtWidgets.QInputDialog') as d:
        d.getItem.return_value = '1', True
        assert pick_item(items, labels) == 'a'
        d.getItem.return_value = '2', True
        assert pick_item(items, labels) == 'b'
        d.getItem.return_value = '3', True
        assert pick_item(items, labels) == 'c'
        d.getItem.return_value = '3', False
        assert pick_item(items, labels) is None


def test_pick_class():

    class Foo:
        pass

    class Bar:
        pass

    Bar.LABEL = 'Baz'

    with mock.patch('glue.utils.qt.dialogs.pick_item') as d:
        pick_class([Foo, Bar], default=Foo)
        d.assert_called_once_with([Foo, Bar], ['Foo', 'Baz'], default=Foo)

    with mock.patch('glue.utils.qt.dialogs.pick_item') as d:
        pick_class([Foo, Bar], sort=True)
        d.assert_called_once_with([Bar, Foo], ['Baz', 'Foo'])


def test_get_text():

    with mock.patch('qtpy.QtWidgets.QInputDialog') as d:

        d.getText.return_value = 'abc', True
        assert get_text() == 'abc'

        d.getText.return_value = 'abc', False
        assert get_text() is None
