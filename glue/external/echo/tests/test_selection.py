import pytest
from unittest.mock import MagicMock

from ..core import HasCallbackProperties, delay_callback
from ..selection import SelectionCallbackProperty, ChoiceSeparator


class Example(HasCallbackProperties):
    a = SelectionCallbackProperty()
    b = SelectionCallbackProperty(default_index=1)
    c = SelectionCallbackProperty(default_index=-1)


class TestSelectionCallbackProperty():

    def setup_method(self, method):
        self.state = Example()
        self.a = Example.a
        self.b = Example.b
        self.c = Example.c

    def test_set_choices(self):
        # make sure that default_index is respected
        self.a.set_choices(self.state, [1, 2])
        self.b.set_choices(self.state, [1, 2])
        self.c.set_choices(self.state, [1, 2, 3])
        assert self.state.a == 1
        assert self.state.b == 2
        assert self.state.c == 3
        self.a.set_choices(self.state, None)
        assert self.state.a is None

    def test_set_value(self):

        # Since there are no choices set yet, the properties cannot be set to
        # any values yet.
        with pytest.raises(ValueError) as exc:
            self.state.a = 1
        assert exc.value.args[0] == "value 1 is not in valid choices: []"

        self.a.set_choices(self.state, [1, 2])

        # The following should work since it is a valid choice
        self.state.a = 2

        # Again if we try and set to an invalid value, things break
        with pytest.raises(ValueError) as exc:
            self.state.a = 3
        assert exc.value.args[0] == "value 3 is not in valid choices: [1, 2]"

    def test_value_constant(self):

        # Make sure that the value is preserved if possible

        self.a.set_choices(self.state, [1, 2])
        self.state.a = 2

        self.a.set_choices(self.state, [2, 5, 3])
        assert self.state.a == 2

        self.a.set_choices(self.state, [1, 4, 5])
        assert self.state.a == 1

        self.a.set_choices(self.state, [4, 2, 1])
        assert self.state.a == 1

    def test_get_choices(self):

        self.a.set_choices(self.state, [1, 2])
        self.a.get_choices == [1, 2]

        self.a.set_choices(self.state, [2, 5, 3])
        self.a.get_choices == [2, 5, 3]

    def test_display_func(self):

        separator = ChoiceSeparator('header')

        self.a.set_choices(self.state, [separator, 1, 2])
        self.a.get_choice_labels(self.state) == ['header', '1', '2']
        assert self.a.get_display_func(self.state) is None
        assert self.b.get_display_func(self.state) is None

        def val(x):
            return 'val{0}'.format(x)

        self.a.set_display_func(self.state, val)
        self.a.get_choice_labels(self.state) == ['header', 'val1', 'val2']
        assert self.a.get_display_func(self.state) is val
        assert self.b.get_display_func(self.state) is None

    def test_callbacks(self):

        # Make sure that callbacks are called when either choices or selection
        # are changed

        func = MagicMock()
        self.state.add_callback('a', func)

        self.a.set_choices(self.state, [1, 2, 3])
        assert func.called_once_with(1)

        self.state.a = 2
        assert func.called_once_with(2)

        self.a.set_choices(self.state, [4, 5, 6])
        assert func.called_once_with(4)

    def test_choice_separator(self):

        separator = ChoiceSeparator('header')
        self.a.set_choices(self.state, [separator, 1, 2])

        assert self.state.a == 1

        separator = ChoiceSeparator('header')
        self.a.set_choices(self.state, [separator])

        assert self.state.a is None

    def test_delay(self):

        func = MagicMock()
        self.state.add_callback('a', func)

        # Here we set the choices and as a result the selection changes from
        # None to a value, so the callback is called after the delay block
        with delay_callback(self.state, 'a'):
            self.a.set_choices(self.state, [4, 5, 6])
            assert func.call_count == 0
        assert func.called_once_with(4)
        func.reset_mock()

        # Check that the callback gets called even if only the choices
        # but not the selection are changed in a delay block
        with delay_callback(self.state, 'a'):
            self.a.set_choices(self.state, [1, 2, 4])
            assert func.call_count == 0
        assert func.called_once_with(4)
        func.reset_mock()
