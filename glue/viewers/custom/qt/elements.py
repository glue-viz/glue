from echo import CallbackProperty, SelectionCallbackProperty

from qtpy.QtWidgets import (QSlider, QLineEdit, QComboBox, QWidget,
                            QLabel, QHBoxLayout, QCheckBox)
from qtpy.QtCore import Qt

__all__ = ["FormElement", "NumberElement", "TextBoxElement", "FloatElement",
           "BoolElement", "ChoiceElement", "FixedComponentIDProperty",
           "FixedComponentElement", "ComponenentElement", "DynamicComponentIDProperty"]


class QLabeledSlider(QWidget):
    """
    A labeled slider widget
    """

    range = None
    integer = None

    def __init__(self, parent=None):

        super(QLabeledSlider, self).__init__(parent)

        self._range = range

        self._slider = QSlider()
        self._slider.setMinimum(0)
        self._slider.setMaximum(100)
        self._slider.setOrientation(Qt.Horizontal)

        self._label = QLabel('')
        self._layout = QHBoxLayout()
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.addWidget(self._slider)
        self._layout.addWidget(self._label)

        self._slider.valueChanged.connect(self._update_label)

        self.setLayout(self._layout)

    def _update_label(self, *args):
        self._label.setText(str(self.value()))

    @property
    def valueChanged(self):
        return self._slider.valueChanged

    def value(self, layer=None, view=None):
        value = self._slider.value() / 100. * (self.range[1] - self.range[0]) + self.range[0]
        if self.integer:
            return int(value)
        else:
            return value

    _in_set_value = False

    def setValue(self, value):
        if self._in_set_value:
            return
        self._in_set_value = True
        value = int(100 * (value - self.range[0]) / (self.range[1] - self.range[0]))
        self._slider.setValue(value)
        self._in_set_value = False


class FormElement(object):

    """
    Base class for user-defined settings in a custom widget.

    Each form element has a value() and a widget. Subclasses
    must override _build_ui, value, and recognizes. They
    may override register_to_hub and add_data.
    """

    def __init__(self, params):
        self.params = params

    @classmethod
    def recognizes(cls, params):
        """
        Returns whether or not a shorthand "params" object
        can be passed to __init__ to construct an element
        """
        raise NotImplementedError

    def ui_and_state(self):
        """
        Build and return a widget to represent this setting.
        """
        raise NotImplementedError()

    @staticmethod
    def auto(params):
        """
        Construct the appropriate FormElement subclass,
        given a shorthand object. For examle,
        FormElement.auto((0., 1.)) returns a NumberElement
        """

        def subclasses(cls):
            return cls.__subclasses__() + [g for s in cls.__subclasses__() for g in subclasses(s)]

        for cls in subclasses(FormElement):
            if cls.recognizes(params):
                return cls(params)
        raise ValueError("Unrecognzied UI Component: %s" % (params,))


class NumberElement(FormElement):

    """
    A form element representing a number

    The shorthand is a tuple of 2 or 3 numbers:
    (min, max) or (min, max default)::

        e = FormElement.auto((0., 1.))
    """

    @classmethod
    def recognizes(cls, params):
        try:
            if len(params) not in [2, 3]:
                return False
            return all(isinstance(p, (int, float)) for p in params)
        except TypeError:
            return False

    def ui_and_state(self):

        if len(self.params) == 3:
            default = self.params[2]
        else:
            default = 0.5 * (self.params[0] + self.params[1])

        # We can't initialize QLabeledSlider yet because this could get called
        # before the Qt application has been initialized. So for now we just make
        # a subclass of QLabeledSlider with the range we need
        class CustomSlider(QLabeledSlider):
            range = self.params[:2]
            integer = isinstance(self.params[0], int) and isinstance(self.params[1], int)

        return 'value_', CustomSlider, CallbackProperty(default)


class TextBoxElement(FormElement):
    """
    A form element representing a generic textbox

    The shorthand is any string starting with an _.::

        e = FormElement.auto("_default")

    Everything after the underscore is taken as the default value.
    """

    @classmethod
    def recognizes(cls, params):
        try:
            if isinstance(params, str) & params.startswith('_'):
                return True
        except AttributeError:
            return None

    def ui_and_state(self):
        default = self.params[1:]
        return 'text_', QLineEdit, CallbackProperty(default)


class FloatElement(FormElement):
    """
    A form element representing a generic number box.

    The shorthand is any number::

        e = FormElement.auto(2)

    The number itself is taken as the default value.
    """

    @classmethod
    def recognizes(cls, params):
        return isinstance(params, (int, float)) and not isinstance(params, bool)

    def ui_and_state(self):
        default = self.params
        return 'valuetext_', QLineEdit, CallbackProperty(default)


class BoolElement(FormElement):

    """
    A checkbox representing a boolean setting

    The shorthand notation is True or False::

        e = FormElement.auto(False)
    """

    @classmethod
    def recognizes(cls, params):
        return isinstance(params, bool)

    def ui_and_state(self):
        default = self.params
        return 'bool_', QCheckBox, CallbackProperty(default)


class ChoiceElement(FormElement):

    """
    A dropdown selector to choose between a set of items

    Shorthand notation is a sequence of strings or a dict::

        e = FormElement.auto({'a':1, 'b':2})
        e = FormElement.auto(['a', 'b', 'c'])
    """

    @classmethod
    def recognizes(cls, params):
        if isinstance(params, str):
            return False
        try:
            return all(isinstance(p, str) for p in params)
        except TypeError:
            return False

    def ui_and_state(self):
        if isinstance(self.params, list):
            choices = self.params
            display_func = None
        else:
            params_inv = dict((value, key) for key, value in self.params.items())
            choices = list(params_inv.keys())
            display_func = params_inv.get
        property = SelectionCallbackProperty(default_index=0, choices=choices,
                                             display_func=display_func)
        return 'combosel_', QComboBox, property


class FixedComponentIDProperty(CallbackProperty):
    pass


class FixedComponentElement(FormElement):

    """
    An element for a Data Component. Does not have a widget

    The shorthand notation is 'att(comp_name)'::

        e = FormElement.auto('att(foo)')
    """

    @classmethod
    def recognizes(cls, params):
        try:
            return params.startswith('att(')
        except AttributeError:
            return False

    def ui_and_state(self):
        component_name = self.params.split('(')[-1][:-1]
        return None, None, FixedComponentIDProperty(component_name)


class DynamicComponentIDProperty(SelectionCallbackProperty):
    pass


class ComponenentElement(FormElement):

    """
    A dropdown selector to choose a component

    The shorthand notation is 'att'::

        e = FormElement.auto('att')
    """

    @classmethod
    def recognizes(cls, params):
        return params == 'att'

    def ui_and_state(self):
        return 'combosel_', QComboBox, DynamicComponentIDProperty()
