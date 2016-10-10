from __future__ import absolute_import, division, print_function

import pytest
from mock import MagicMock

from glue.core import ComponentID
from glue.config import link_function, link_helper

from ..link_equation import (function_label, helper_label,
                             LinkEquation, ArgumentWidget)


@link_function('testing function', ['y'])
def func1(x):
    return x


@link_function('testing function', ['a', 'b'])
def func2(x, z):
    return x + z, x - z


@link_helper('test helper', ['a', 'b'])
def helper(x, y):
    return [x, x, y]


def test_function_label():
    f1 = [f for f in link_function if f[0] is func1][0]
    f2 = [f for f in link_function if f[0] is func2][0]
    assert function_label(f1) == "Link from x to y"
    assert function_label(f2) == "Link from x, z to a, b"


def test_helper_label():
    f1 = [h for h in link_helper if h[0] is helper][0]
    assert helper_label(f1) == 'test helper'


class TestArgumentWidget(object):
    def test_label(self):
        a = ArgumentWidget('test')
        assert a.label == 'test'

    def test_set_label(self):
        a = ArgumentWidget('test')
        a.label = 'b'
        assert a.label == 'b'

    def test_drop(self):
        target_id = ComponentID('test')
        event = MagicMock()
        event.mimeData().data.return_value = target_id
        a = ArgumentWidget('test')
        a.dropEvent(event)
        assert a.component_id is target_id
        assert a.editor_text == 'test'

    def test_drop_invalid(self):
        event = MagicMock()
        event.mimeData().data.return_value = 5
        a = ArgumentWidget('')
        a.dropEvent(event)
        assert a.component_id is None

    def test_clear(self):
        target_id = ComponentID('test')
        event = MagicMock()
        event.mimeData().data.return_value = target_id
        a = ArgumentWidget('test')
        a.dropEvent(event)
        assert a.component_id is target_id
        a.clear()
        assert a.component_id is None
        assert a.editor_text == ''

    def test_drag_enter_accept(self):
        event = MagicMock()
        event.mimeData().hasFormat.return_value = True
        a = ArgumentWidget('x')
        a.dragEnterEvent(event)
        event.accept.assert_called_once_with()

    def test_drag_enter_ignore(self):
        event = MagicMock()
        event.mimeData().hasFormat.return_value = False
        a = ArgumentWidget('x')
        a.dragEnterEvent(event)
        event.ignore.assert_called_once_with()


class TestLinkEquation(object):
    def setup_method(self, method):
        self.widget = LinkEquation()

    def test_select_function_member(self):
        member = link_function.members[1]
        assert self.widget.function is not member
        self.widget.function = member
        assert self.widget.function is member

    def test_select_function_helper(self):
        member = link_helper.members[-1]
        self.widget.function = member
        assert self.widget.function is member

    def test_select_invalid_function(self):
        with pytest.raises(ValueError) as exc:
            def bad(x):
                pass
            self.widget.function = (bad, None, None)
        assert exc.value.args[0].startswith('Cannot find data')

    def test_make_link_function(self):
        widget = LinkEquation()
        f1 = [f for f in link_function if f[0] is func1][0]
        widget.function = f1
        x, y = ComponentID('x'), ComponentID('y')
        widget.signature = [x], y
        links = widget.links()
        assert len(links) == 1
        assert links[0].get_from_ids() == [x]
        assert links[0].get_to_id() == y
        assert links[0].get_using() is func1

    def test_make_link_helper(self):
        widget = LinkEquation()
        f1 = [f for f in link_helper if f[0] is helper][0]
        widget.function = f1
        x, y = ComponentID('x'), ComponentID('y')
        widget.signature = [x, y], None
        links = widget.links()
        assert links == helper(x, y)

    def test_links_empty(self):
        assert LinkEquation().links() == []

    def test_links_empty_helper(self):
        widget = LinkEquation()
        f1 = [f for f in link_helper if f[0] is helper][0]
        widget.function = f1
        assert widget.is_helper()
        assert widget.links() == []

    def test_clear_inputs(self):
        widget = LinkEquation()
        f1 = [f for f in link_helper if f[0] is helper][0]
        widget.function = f1
        x, y = ComponentID('x'), ComponentID('y')

        widget.signature = [x, y], None
        assert widget.signature == ([x, y], None)

        widget.clear_inputs()
        assert widget.signature == ([None, None], None)

    def test_signal_connections(self):
        # testing that signal-slot connections don't crash
        widget = LinkEquation()

        signal = widget._ui.function.currentIndexChanged
        signal.emit(5)

        signal = widget._output_widget.editor.textChanged
        signal.emit('changing')
