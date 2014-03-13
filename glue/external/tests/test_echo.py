from mock import MagicMock
import pytest

from ..echo import (CallbackProperty, add_callback,
                    remove_callback, delay_callback,
                    ignore_callback, callback_property)


class Stub(object):
    prop1 = CallbackProperty()
    prop2 = CallbackProperty(5)
    prop3 = 5


class DecoratorStub(object):

    def __init__(self):
        self._val = 1

    @callback_property
    def prop(self):
        return self._val * 2

    @prop.setter
    def prop(self, value):
        self._val = value


def test_attribute_like_access():
    stub = Stub()
    assert stub.prop1 is None
    assert stub.prop2 == 5


def test_attribute_like_set():
    stub = Stub()
    stub.prop1 = 10
    assert stub.prop1 == 10


def test_class_access():
    stub = Stub()
    assert isinstance(type(stub).prop1, CallbackProperty)


def test_callback_fire_on_change():
    stub = Stub()
    test = MagicMock()
    add_callback(stub, 'prop1', test)
    stub.prop1 = 5
    test.assert_called_once_with(5)


def test_callbacks_only_called_on_value_change():
    stub = Stub()
    test = MagicMock()
    add_callback(stub, 'prop1', test)
    stub.prop1 = 5
    test.assert_called_once_with(5)
    stub.prop1 = 5
    assert test.call_count == 1


def test_callbacks_are_instance_specific():
    s1, s2 = Stub(), Stub()
    test = MagicMock()
    add_callback(s2, 'prop1', test)
    s1.prop1 = 100
    assert test.call_count == 0


def test_remove_callback():
    stub = Stub()
    test = MagicMock()
    add_callback(stub, 'prop1', test)
    remove_callback(stub, 'prop1', test)
    stub.prop1 = 5
    assert test.call_count == 0


def test_add_callback_attribute_error_on_bad_name():
    stub = Stub()
    with pytest.raises(AttributeError):
        add_callback(stub, 'bad_property', None)


def test_add_callback_type_error_if_not_calllback():
    stub = Stub()
    with pytest.raises(TypeError) as exc:
        add_callback(stub, 'prop3', None)
    assert exc.value.args[0] == "prop3 is not a CallbackProperty"


def test_remove_callback_attribute_error_on_bad_name():
    stub = Stub()
    with pytest.raises(AttributeError):
        remove_callback(stub, 'bad_property', None)


def test_remove_callback_wrong_function():
    stub = Stub()
    test = MagicMock()
    test2 = MagicMock()
    add_callback(stub, 'prop1', test)
    with pytest.raises(ValueError) as exc:
        remove_callback(stub, 'prop1', test2)
    assert exc.value.args[0].startswith('Callback function not found')


def test_remove_non_callback_property():
    stub = Stub()
    with pytest.raises(TypeError) as exc:
        remove_callback(stub, 'prop3', None)
    assert exc.value.args[0] == 'prop3 is not a CallbackProperty'


def test_remove_callback_not_found():
    stub = Stub()
    with pytest.raises(ValueError) as exc:
        remove_callback(stub, 'prop1', None)
    assert exc.value.args[0] == "Callback function not found: None"


def test_disable_callback():
    stub = Stub()
    test = MagicMock()

    add_callback(stub, 'prop1', test)
    Stub.prop1.disable(stub)

    stub.prop1 = 100
    assert test.call_count == 0

    Stub.prop1.enable(stub)
    stub.prop1 = 100
    assert test.call_count == 0  # not changed
    stub.prop1 = 200
    assert test.call_count == 1


def test_delay_callback():
    test = MagicMock()
    stub = Stub()

    add_callback(stub, 'prop1', test)
    with delay_callback(stub, 'prop1'):
        stub.prop1 = 100
        stub.prop1 = 200
        stub.prop1 = 300
        assert test.call_count == 0
    test.assert_called_once_with(300)


def test_delay_callback_not_called_if_unmodified():
    test = MagicMock()
    stub = Stub()
    add_callback(stub, 'prop1', test)
    with delay_callback(stub, 'prop1'):
        pass
    assert test.call_count == 0


def test_callback_with_two_arguments():
    stub = Stub()
    stub.prop1 = 5
    on_change = MagicMock()
    add_callback(stub, 'prop1', on_change, echo_old=True)
    stub.prop1 = 10

    on_change.assert_called_once_with(5, 10)


@pytest.mark.parametrize('context_func', (delay_callback, ignore_callback))
def test_context_on_non_callback(context_func):
    stub = Stub()
    with pytest.raises(TypeError) as exc:
        with context_func(stub, 'prop3'):
            pass
    assert exc.value.args[0] == "prop3 is not a CallbackProperty"


def test_delay_multiple():
    stub = Stub()
    test = MagicMock()
    test2 = MagicMock()

    add_callback(stub, 'prop1', test)
    add_callback(stub, 'prop2', test2)

    with delay_callback(stub, 'prop1', 'prop2'):
        stub.prop1 = 50
        stub.prop1 = 100
        stub.prop2 = 200
        assert test.call_count == 0
        assert test2.call_count == 0

    test.assert_called_once_with(100)
    test2.assert_called_once_with(200)


def test_ignore_multiple():
    stub = Stub()
    test = MagicMock()
    test2 = MagicMock()

    add_callback(stub, 'prop1', test)
    add_callback(stub, 'prop2', test2)

    with ignore_callback(stub, 'prop1', 'prop2'):
        stub.prop1 = 100
        stub.prop2 = 200
        assert test.call_count == 0
        assert test2.call_count == 0

    assert test.call_count == 0
    assert test2.call_count == 0


def test_delay_only_calls_if_changed():
    stub = Stub()
    test = MagicMock()

    add_callback(stub, 'prop1', test)

    with delay_callback(stub, 'prop1'):
        pass
    assert test.call_count == 0

    val = stub.prop1
    with delay_callback(stub, 'prop1'):
        stub.prop1 = val
    assert test.call_count == 0


def test_decorator_form():
    stub = DecoratorStub()
    test = MagicMock()
    add_callback(stub, 'prop', test)

    assert stub.prop == 2

    stub.prop = 5
    test.assert_called_once_with(10)

    assert stub.prop == 10
