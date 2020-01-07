import pytest

from glue.core.state_objects import State
from ..elements import (FormElement, NumberElement, ChoiceElement,
                        FloatElement, TextBoxElement)


def get_value(element):

    prefix, widget_cls, property = element.ui_and_state()

    class TemporaryState(State):
        a = property

    temp = TemporaryState()
    return temp.a


class TestFormElements(object):

    def test_number_default_value(self):
        e = FormElement.auto((0, 100, 30))
        assert get_value(e) == 30

    def test_number_float(self):
        e = FormElement.auto((0.0, 1.0, 0.3))
        assert get_value(e) == 0.3

    def test_number_list(self):
        e = FormElement.auto([0, 10])
        assert isinstance(e, NumberElement)

    def test_choice_list(self):
        e = FormElement.auto(['a', 'b'])
        assert isinstance(e, ChoiceElement)

    def test_choice_tuple(self):
        e = FormElement.auto(('a', 'b'))
        assert isinstance(e, ChoiceElement)

    def test_float(self):
        e = FormElement.auto(1.2)
        assert isinstance(e, FloatElement)

        e = FormElement.auto(2)
        assert isinstance(e, FloatElement)
        assert get_value(e) == 2

    def test_textbox(self):
        e = FormElement.auto('_str')
        assert isinstance(e, TextBoxElement)
        assert get_value(e) == 'str'

    def test_recognizes_subsubclasses(self):

        class SubClassFormElement(TextBoxElement):
            @classmethod
            def recognizes(cls, params):
                return params == 'specific_class'

        e = FormElement.auto('specific_class')
        assert isinstance(e, SubClassFormElement)

    def test_unrecognized(self):
        with pytest.raises(ValueError):
            FormElement.auto(None)
