from mock import MagicMock
import pytest

from ..callback_property import (CallbackProperty, add_callback,
                                 remove_callback, delay_callback,
                                 ignore_callback)


class TestClass(object):
    prop1 = CallbackProperty()
    prop2 = CallbackProperty(5)
    prop3 = 5


def test_transparent_access():
    tc = TestClass()
    assert tc.prop1 is None
    assert tc.prop2 == 5


def test_transparent_set():
    tc = TestClass()
    tc.prop1 = 10
    assert tc.prop1 == 10


def test_class_access():
    tc = TestClass()
    assert isinstance(type(tc).prop1, CallbackProperty)


def test_add_callback():
    tc = TestClass()
    test = MagicMock()
    add_callback(tc, 'prop1', test)
    tc.prop1 = 5
    test.assert_called_once_with(5)


def test_callbacks_only_called_on_value_change():
    tc = TestClass()
    test = MagicMock()
    add_callback(tc, 'prop1', test)
    tc.prop1 = 5
    test.assert_called_once_with(5)
    tc.prop1 = 5
    assert test.call_count == 1


def test_remove_callback():
    tc = TestClass()
    test = MagicMock()
    add_callback(tc, 'prop1', test)
    remove_callback(tc, 'prop1', test)
    tc.prop1 = 5
    assert test.call_count == 0


def test_add_callback_attribute_error():
    tc = TestClass()
    with pytest.raises(AttributeError) as exc:
        add_callback(tc, 'bad_property', None)


def test_add_callback_type_error():
    tc = TestClass()
    with pytest.raises(TypeError) as exc:
        add_callback(tc, 'prop3', None)
    assert exc.value.args[0] == "prop3 is not a CallbackProperty"


def test_remove_callback_attribute_error():
    tc = TestClass()
    with pytest.raises(AttributeError) as exc:
        remove_callback(tc, 'bad_property', None)


def test_add_callback_type_error():
    tc = TestClass()
    with pytest.raises(TypeError) as exc:
        remove_callback(tc, 'prop3', None)
    assert exc.value.args[0] == "prop3 is not a CallbackProperty"


def test_remove_callback_not_found():
    tc = TestClass()
    with pytest.raises(ValueError) as exc:
        remove_callback(tc, 'prop1', None)
    assert exc.value.args[0] == "Callback function not found: None"


def test_disable_callback():
    tc = TestClass()
    test = MagicMock()

    add_callback(tc, 'prop1', test)
    TestClass.prop1.disable(tc)

    tc.prop1 = 100
    assert test.call_count == 0

    TestClass.prop1.enable(tc)
    tc.prop1 = 100
    assert test.call_count == 0  # not changed
    tc.prop1 = 200
    assert test.call_count == 1


def test_delay_callback():
    test = MagicMock()
    tc = TestClass()

    add_callback(tc, 'prop1', test)
    with delay_callback(tc, 'prop1'):
        tc.prop1 = 100
        tc.prop1 = 200
        tc.prop1 = 300
        assert test.call_count == 0
    test.assert_called_once_with(300)


def test_delay_type_error():
    tc = TestClass()
    with pytest.raises(TypeError) as exc:
        with delay_callback(tc, 'prop3'):
            pass
    assert exc.value.args[0] == "prop3 is not a CallbackProperty"


def test_delay_multiple():
    tc = TestClass()
    test = MagicMock()
    test2 = MagicMock()

    add_callback(tc, 'prop1', test)
    add_callback(tc, 'prop2', test2)

    with delay_callback(tc, 'prop1', 'prop2'):
        tc.prop1 = 100
        tc.prop2 = 200
        assert test.call_count == 0
        assert test2.call_count == 0

    test.assert_called_once_with(100)
    test2.assert_called_once_with(200)


def test_ignore_multiple():
    tc = TestClass()
    test = MagicMock()
    test2 = MagicMock()

    add_callback(tc, 'prop1', test)
    add_callback(tc, 'prop2', test2)

    with ignore_callback(tc, 'prop1', 'prop2'):
        tc.prop1 = 100
        tc.prop2 = 200
        assert test.call_count == 0
        assert test2.call_count == 0

    assert test.call_count == 0
    assert test2.call_count == 0


def test_delay_only_calls_if_changed():
    tc = TestClass()
    test = MagicMock()

    add_callback(tc, 'prop1', test)

    with delay_callback(tc, 'prop1'):
        pass
    assert test.call_count == 0

    val = tc.prop1
    with delay_callback(tc, 'prop1'):
        tc.prop1 = val
    assert test.call_count == 0
